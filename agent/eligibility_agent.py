import json
import os
import re
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_action_plan(profile: dict, candidate_schemes: list) -> list:
    """
    Takes user profile + candidate schemes
    Returns top 3-5 schemes with eligibility verdict and action plan
    """
    language = profile.get("language", "English")

    scheme_summaries = []
    for s in candidate_schemes:
        scheme_summaries.append({
            "name": s.get("name"),
            "eligibility": s.get("eligibility", "")[:400],
            "benefits": s.get("benefits", "")[:300],
            "documents": s.get("documents_required", [])[:5],
            "how_to_apply": s.get("application_process", "")[:200],
            "source_url": s.get("source_url", "")
        })

    prompt = f"""You are a government scheme eligibility expert for India.

User Profile:
{json.dumps(profile, indent=2)}

Candidate Schemes:
{json.dumps(scheme_summaries, indent=2)}

Task:
Evaluate eligibility and return ONLY a JSON array. No other text.
Respond in {language}.

[
  {{
    "scheme_name": "...",
    "eligible": true,
    "why_eligible": "1-2 lines explaining why in {language}",
    "what_you_get": "Key benefit in {language}",
    "documents_needed": ["doc1", "doc2", "doc3"],
    "how_to_apply": "Simple 2-3 step process in {language}",
    "source_url": "..."
  }}
]

Only include schemes where eligible is true.
Maximum 5 schemes.
Use simple language a rural citizen would understand.
Output JSON array only."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=3000
    )

    text = response.choices[0].message.content.strip()
    text = re.sub(r"```json|```", "", text).strip()

    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        return json.loads(text[start:end])
    except Exception as e:
        print(f"Parse error: {e}")
        print(f"Raw response: {text[:500]}")
        return []