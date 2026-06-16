import re
from typing import List, Dict, Any

INJECTION_KEYWORDS = [
    r"ignore previous instructions",
    r"system prompt",
    r"reveal your secrets",
    r"act as a malicious",
    r"you are now an unrestricted"
]

def check_prompt_injection(query: str) -> bool:
    for pattern in INJECTION_KEYWORDS:
        if re.search(pattern, query.lower()):
            return True
    return False

def basic_hallucination_check(response: str, context_docs: List[Dict[str, Any]]) -> float:
    if not context_docs:
        return 1.0
    combined_context = " ".join([doc["content"].lower() for doc in context_docs])
    context_words = set(re.findall(r'\b\w{4,}\b', combined_context))
    if not context_words:
        return 1.0
    response_words = re.findall(r'\b\w{4,}\b', response.lower())
    if not response_words:
        return 1.0
    matched_words = [word for word in response_words if word in context_words]
    return len(matched_words) / len(response_words)
