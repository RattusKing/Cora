DISCLAIMER = (
    "Research-only assistant focused on marine genomics and literature. "
    "Not medical advice. Not a lab protocol. No actionable experiments."
)

REFUSAL = (
    "I can’t help with experimental protocols, step-by-step lab instructions, "
    "or anything that enables biological manipulation. I can summarize literature "
    "and discuss high-level concepts with citations."
)

def check_safety(user_text: str) -> bool:
    t = user_text.lower()
    return any(k in t for k in DISALLOWED_KEYWORDS)
