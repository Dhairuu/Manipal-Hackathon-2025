from fastapi import FastAPI
from pydantic import BaseModel, Field
import ollama
import json
import re

app = FastAPI()

class SuggestionRequest(BaseModel):
    merchant: str
    description: str
    category: str
    subcategory: str
    goal: str
    income: int = Field(gt=0, description="Monthly income must be greater than 0")

def suggest_goal_aligned_alternative(merchant, description, category, subcategory, goal, income):
    prompt = f"""
    A user made the following transaction:

    Description: "{description}"
    Merchant: {merchant}
    Category: {category}
    Subcategory: {subcategory}
    Monthly Income: ₹{income}

    The user has a financial goal: "{goal}"

    Your task:
    1. Analyze the spending in the description.
    2. Suggest a **cheaper or more essential alternative** to this transaction.
    3. Quantify the potential savings (e.g., 'you could save ₹300 per week').
    4. Explain **how this change helps the user reach their goal faster**, in a friendly and motivating way.
    5. Make sure the recommendation is appropriate for someone with this income.

    Be concise but insightful.
    """

    response = ollama.chat(
        model="gemma3:1b",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response["message"]["content"].strip()

def beautifyJson(text):
    prompt = f"""
    The following is a response split into four sections. Please convert it into a JSON array where each section becomes an object with two keys: "title" and "description".

    Response:
    {text}

    Return the result as a JSON array of objects like:
    [
      {{
        "title": "...",
        "description": "..."
      }},
      ...
    ]
    """

    response = ollama.chat(
        model="gemma3:1b",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response["message"]["content"].strip()

def safe_parse_json(text):
    try:
        text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")

        start_idx = text.find('[')
        if start_idx == -1:
            raise ValueError("No JSON array start found")

        end_idx = text.find('```', start_idx)
        if end_idx == -1:
            end_idx = len(text)

        json_string = text[start_idx:end_idx].strip()

        json_string = json_string.replace('\x00', '').replace('\r', '')

        return json.loads(json_string)

    except (json.JSONDecodeError, ValueError) as e:
        print("❌ JSON parsing failed:", e)
        print("⛔ Raw string:", text)
        return None

@app.post("/suggest-alternate")
def suggest_alternate_endpoint(request: SuggestionRequest):
    raw_response = suggest_goal_aligned_alternative(
        request.merchant,
        request.description,
        request.category,
        request.subcategory,
        request.goal,
        request.income
    )

    beautified = beautifyJson(raw_response)
    parsed = safe_parse_json(beautified)


    return parsed if parsed else {
        "error": "Failed to parse beautified JSON",
        "raw": beautified
    }
