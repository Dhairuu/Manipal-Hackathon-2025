from fastapi import FastAPI, Query
from pydantic import BaseModel
import json
import os
from dotenv import load_dotenv
import ollama
from tavily import TavilyClient

load_dotenv()
app = FastAPI()

# Load MCC data once
with open("final.json") as f:
    mcc_data = json.load(f)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class MerchantRequest(BaseModel):
    merchant: str

def fetch_merchant_description(merchant):
    try:
        result = tavily_client.search(
            query=merchant,
            search_depth="advanced",
            include_answer=True,
            max_results=5
        )
        return result.get("answer", "")
    except Exception as e:
        print(f"[Tavily Error] {e}")
        return ""

def ask_llm_for_general_category(description):
    general_categories = ', '.join([entry for entry in mcc_data])
    prompt = f"""
    You are a merchant category assistant. From the given category division choose the most appropriate category for the user's input. Incase there appears to be no match an additional others category is given return that.
    Merchant Description:
    \"\"\"{description}\"\"\"

    Valid Categories:
    {general_categories}

    Respond with only the category name from the valid Categories list given.
    """
    response = ollama.chat(
        model="gemma3:1b",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response["message"]["content"].strip()

def ask_llm_for_subcategory(merchant, description, general_category):
    categories = ', '.join([entry for entry in mcc_data[general_category]])
    prompt = f"""
        A merchant named {merchant} falls under the broader category of {general_category} for the description {description}.

        Now choose the **most appropriate subcategory** from the list below that best describes this merchant's core business **strictly choose from the valid subcategories**, if the broader category is Other just return Unknown:
        Valid categories: {categories}

        Respond with only one subcategory name from the valid subcategories list given.
    """
    response = ollama.chat(
        model="gemma3:1b",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response["message"]["content"].strip()

@app.post("/classify")
async def classify_merchant(data: MerchantRequest):
    merchant = data.merchant
    description = fetch_merchant_description(merchant)
    general_category = ask_llm_for_general_category(description)

    if general_category == "Other":
        return {
            "merchant": merchant,
            "description": description,
            "general_category": "Other",
            "subcategory": "Unknown",
            "details": {
                "code": 0,
                "description": "Unknown category",
                "need": "Unidentified"
            }
        }
    
    subcategory = ask_llm_for_subcategory(merchant, description, general_category)
    subcategory_data = mcc_data[general_category].get(subcategory, {
        "code": 0,
        "description": "Unknown category",
        "need": "Unidentified"
    })

    return {
        "merchant": merchant,
        "description": description,
        "general_category": general_category,
        "subcategory": subcategory,
        "details": subcategory_data
    }
