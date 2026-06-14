import json
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are YojnaBot, a friendly assistant helping Indian citizens 
discover government schemes in their own language.

Your job is to collect a user profile through natural conversation.
Required fields:
- state: Indian state they live in
- occupation: farmer, student, woman entrepreneur, daily wage worker, salaried, business, unemployed
- gender: male or female
- category: general, OBC, SC, ST
- income_annual: approximate annual household income in rupees
- age: age in years

Rules:
- Ask ONE question at a time in a friendly conversational way
- If user writes in Hindi, Marathi, Telugu, Bengali or any Indian language — respond in that same language
- Do not ask for fields you already have
- Once ALL fields are collected output ONLY this JSON and nothing else:

{
  "profile_complete": true,
  "state": "...",
  "occupation": "...",
  "gender": "...",
  "category": "...",
  "income_annual": 0,
  "age": 0,
  "language": "Hindi or English or Marathi etc"
}

Never output JSON until every single field is confirmed."""


def extract_profile(conversation_history: list) -> dict:
    """
    Takes conversation history and returns either:
    - {"profile_complete": false, "next_question": "..."}
    - {"profile_complete": true, "state": ..., ...}
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for turn in conversation_history:
        messages.append({
            "role": "user" if turn["role"] == "user" else "assistant",
            "content": turn["content"]
        })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=500
    )

    text = response.choices[0].message.content.strip()

    # Check if profile is complete
    if "{" in text and "profile_complete" in text:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            profile = json.loads(text[start:end])
            if profile.get("profile_complete"):
                return profile
        except:
            pass

    return {
        "profile_complete": False,
        "next_question": text
    }


def start_conversation() -> str:
    return "नमस्ते! मैं YojnaBot हूँ 🙏 मैं आपको सरकारी योजनाएं खोजने में मदद करूँगा। आप किस राज्य में रहते हैं? (Hello! I am YojnaBot. Which state do you live in?)"