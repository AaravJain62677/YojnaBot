import streamlit as st
import json
import os
import re
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# ── Configuration ─────────────────────────────────────────
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PROFILE_PROMPT = """You are YojnaBot, a friendly assistant helping Indian citizens 
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


# ── LLM Helper ────────────────────────────────────────────
def call_llm(messages: list, system_prompt: str = None,
             temperature=0.3, max_tokens=500) -> str:
    formatted = []
    if system_prompt:
        formatted.append({"role": "system", "content": system_prompt})
    for turn in messages:
        formatted.append({
            "role": "user" if turn["role"] == "user" else "assistant",
            "content": turn["content"]
        })

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=formatted,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


# ── Load Data ─────────────────────────────────────────────
@st.cache_data
def load_schemes():
    schemes = []
    with open("data/cleaned_schemes.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            schemes.append(json.loads(line))
    return schemes


@st.cache_data
def load_adapted():
    adapted = {}
    filepath = "data/adapted_eligibility_final.jsonl"
    if not os.path.exists(filepath):
        return adapted
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            scheme_id = obj.get("scheme_id", "")
            if scheme_id:
                adapted[scheme_id] = obj.get("multilingual_eligibility", "")
    return adapted


SCHEMES = load_schemes()
ADAPTED = load_adapted()


# ── Agent 1: Profile Chat ─────────────────────────────────
def chat_profile(history: list) -> str:
    return call_llm(
        messages=history,
        system_prompt=PROFILE_PROMPT,
        temperature=0.3,
        max_tokens=500
    )


def extract_json_profile(text: str) -> dict:
    if "{" in text and "profile_complete" in text:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except:
            pass
    return {}


# ── Agent 2: Scheme Filter ────────────────────────────────
def filter_schemes(profile: dict) -> list:
    state = profile.get("state", "").lower()
    occupation = profile.get("occupation", "").lower()
    gender = profile.get("gender", "").lower()
    category = profile.get("category", "").lower()

    occupation_keywords = {
        "farmer": ["farmer", "agriculture", "kisan", "crop", "farming", "agri"],
        "student": ["student", "education", "scholarship", "school", "college"],
        "woman entrepreneur": ["women", "woman", "mahila", "female", "entrepreneur"],
        "daily wage": ["labour", "worker", "daily wage", "construction"],
        "unemployed": ["unemployed", "unemployment", "job seeker"],
        "salaried": ["employee", "salaried", "government employee"],
        "business": ["business", "entrepreneur", "msme", "startup"],
    }

    category_keywords = {
        "sc": ["scheduled caste", "sc/st", " sc ", "dalit"],
        "st": ["scheduled tribe", "sc/st", "tribal", " st "],
        "obc": ["obc", "other backward", "backward class"],
    }

    candidates = []

    for scheme in SCHEMES:
        score = 0
        eligibility = scheme.get("eligibility", "").lower()
        description = scheme.get("description", "").lower()
        tags = " ".join(scheme.get("tags", [])).lower()
        combined = eligibility + " " + description + " " + tags

        scheme_states = [
            s.lower() for s in scheme.get("beneficiary_state", ["all"])
        ]
        if "all" in scheme_states or state in scheme_states:
            score += 2

        if scheme.get("level") == "Central":
            score += 1

        for occ, keywords in occupation_keywords.items():
            if occ in occupation:
                if any(kw in combined for kw in keywords):
                    score += 3
                break

        if category in category_keywords:
            if any(kw in combined for kw in category_keywords[category]):
                score += 2

        if gender == "female":
            if any(kw in combined for kw in
                   ["women", "woman", "female", "mahila", "girl"]):
                score += 2

        if score > 0:
            candidates.append((score, scheme))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in candidates[:10]]


# ── Agent 3: Action Plan ──────────────────────────────────
def generate_action_plan(profile: dict, candidates: list) -> list:
    language = profile.get("language", "English")

    scheme_summaries = []
    for s in candidates:
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

    text = call_llm(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=3000
    )

    text = re.sub(r"```json|```", "", text).strip()

    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        return json.loads(text[start:end])
    except Exception as e:
        st.error(f"Parse error: {e}")
        return []


# ── Streamlit UI ──────────────────────────────────────────
st.set_page_config(
    page_title="YojnaBot",
    page_icon="🇮🇳",
    layout="centered"
)

st.title("🇮🇳 YojnaBot")
st.caption("Find government schemes you qualify for — in your language")

if "stage" not in st.session_state:
    st.session_state.stage = "chat"
if "history" not in st.session_state:
    st.session_state.history = []
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "results" not in st.session_state:
    st.session_state.results = []
if "started" not in st.session_state:
    st.session_state.started = False

# ── Stage 1: Chat ─────────────────────────────────────────
if st.session_state.stage == "chat":

    if not st.session_state.started:
        opening = "नमस्ते! मैं YojnaBot हूँ 🙏 आप किस राज्य में रहते हैं? (Hello! I am YojnaBot. Which state do you live in?)"
        st.session_state.history.append({
            "role": "model",
            "content": opening
        })
        st.session_state.started = True

    for turn in st.session_state.history:
        if turn["role"] == "model":
            with st.chat_message("assistant"):
                st.write(turn["content"])
        else:
            with st.chat_message("user"):
                st.write(turn["content"])

    user_input = st.chat_input("Type your response...")

    if user_input:
        st.session_state.history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_profile(st.session_state.history)

            profile = extract_json_profile(response)

            if profile.get("profile_complete"):
                st.session_state.profile = profile
                st.write("✅ Got your profile! Finding schemes...")
                st.session_state.history.append({
                    "role": "model",
                    "content": response
                })

                with st.spinner("Matching schemes..."):
                    candidates = filter_schemes(profile)
                    if not candidates:
                        candidates = [
                            s for s in SCHEMES
                            if s.get("level") == "Central"
                        ][:10]

                    results = generate_action_plan(profile, candidates)
                    st.session_state.results = results
                    st.session_state.stage = "results"
                    st.rerun()
            else:
                st.write(response)
                st.session_state.history.append({
                    "role": "model",
                    "content": response
                })

# ── Stage 2: Results ──────────────────────────────────────
elif st.session_state.stage == "results":

    profile = st.session_state.profile
    results = st.session_state.results

    st.success(f"Found {len(results)} schemes for you!")

    with st.expander("Your Profile", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("State", profile.get("state", "-"))
        col2.metric("Occupation", profile.get("occupation", "-"))
        col3.metric("Category", profile.get("category", "-"))

    st.divider()

    if not results:
        st.warning("No matching schemes found. Try adjusting your profile.")
    else:
        for i, scheme in enumerate(results):
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.subheader(f"📋 {scheme.get('scheme_name', 'Scheme')}")
                with col2:
                    if scheme.get("eligible"):
                        st.success("✅ Eligible")

                # Show Adaption multilingual data if available
                scheme_id = next(
                    (s.get("scheme_id") for s in SCHEMES
                     if s.get("name") == scheme.get("scheme_name")),
                    None
                )
                if scheme_id and scheme_id in ADAPTED:
                    with st.expander("🌐 View eligibility in your language"):
                        st.markdown(ADAPTED[scheme_id])

                st.markdown(
                    f"**Why you qualify:** {scheme.get('why_eligible', '')}"
                )
                st.info(f"💰 **Benefit:** {scheme.get('what_you_get', '')}")

                docs = scheme.get("documents_needed", [])
                if docs:
                    st.markdown("**Documents needed:**")
                    for doc in docs:
                        st.markdown(f"- {doc}")

                st.markdown(
                    f"**How to apply:** {scheme.get('how_to_apply', '')}"
                )

                url = scheme.get("source_url", "")
                if url:
                    st.markdown(f"[🔗 View full scheme details]({url})")

                st.markdown("**Was this helpful?**")
                col1, col2 = st.columns([1, 8])
                with col1:
                    if st.button("👎", key=f"feedback_{i}"):
                        feedback = {
                            "scheme_name": scheme.get("scheme_name"),
                            "profile": profile,
                            "issue": "User flagged as incorrect"
                        }
                        with open("data/corrections.jsonl", "a",
                                  encoding="utf-8") as f:
                            f.write(
                                json.dumps(feedback,
                                           ensure_ascii=False) + "\n"
                            )
                        st.toast("Feedback saved. This will improve our dataset.")

                st.divider()

    if st.button("🔄 Start Over"):
        for key in ["stage", "history", "profile", "results", "started"]:
            del st.session_state[key]
        st.rerun()