import json

def load_schemes(filepath="data/cleaned_schemes.jsonl"):
    schemes = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            schemes.append(json.loads(line))
    return schemes

def filter_schemes(profile: dict, schemes: list) -> list:
    """
    Filter schemes based on user profile.
    Returns top 10 candidate schemes for the eligibility agent.
    """
    candidates = []
    
    state = profile.get("state", "").lower()
    occupation = profile.get("occupation", "").lower()
    gender = profile.get("gender", "").lower()
    category = profile.get("category", "").lower()
    age = profile.get("age", 0)
    income = profile.get("income_annual", 0)

    for scheme in schemes:
        score = 0
        
        # State match
        scheme_states = [s.lower() for s in scheme.get("beneficiary_state", ["all"])]
        if "all" in scheme_states or state in scheme_states:
            score += 2

        # Category match from tags and eligibility text
        eligibility = scheme.get("eligibility", "").lower()
        description = scheme.get("description", "").lower()
        combined_text = eligibility + " " + description

        # Occupation match
        occupation_keywords = {
            "farmer": ["farmer", "agriculture", "kisan", "crop", "farming"],
            "student": ["student", "education", "scholarship", "school", "college"],
            "woman": ["women", "woman", "mahila", "female", "girl"],
            "entrepreneur": ["entrepreneur", "business", "startup", "msme"],
            "daily wage": ["labour", "worker", "daily wage", "construction"],
        }
        
        for occ, keywords in occupation_keywords.items():
            if occ in occupation:
                if any(kw in combined_text for kw in keywords):
                    score += 3

        # Category match
        category_keywords = {
            "sc": ["scheduled caste", "sc/st", "sc ", "dalit"],
            "st": ["scheduled tribe", "sc/st", "tribal", "st "],
            "obc": ["obc", "other backward"],
            "general": []
        }
        
        if category in category_keywords:
            if any(kw in combined_text for kw in category_keywords[category]):
                score += 2

        # Gender match
        if gender == "female":
            if any(kw in combined_text for kw in ["women", "woman", "female", "girl", "mahila"]):
                score += 2

        # Always include central schemes with some relevance
        if scheme.get("level") == "Central" and score > 0:
            score += 1

        if score > 0:
            candidates.append((score, scheme))

    # Sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Return top 10
    return [scheme for score, scheme in candidates[:10]]


def get_fallback_schemes(schemes: list, n=5) -> list:
    """Return popular central schemes if no matches found"""
    central = [s for s in schemes if s.get("level") == "Central"]
    return central[:n]