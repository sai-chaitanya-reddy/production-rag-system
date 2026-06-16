import re
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

# Simple rule-based patterns for malicious injection defense
INJECTION_KEYWORDS = [
    r"ignore previous instructions", 
    r"system prompt", 
    r"reveal your secrets", 
    r"act as a malicious",
    r"you are now an unrestricted"
]

def check_prompt_injection(query: str) -> bool:
    """Returns True if malicious intent or system hijack behavior is observed in the prompt."""
    for pattern in INJECTION_KEYWORDS:
        if re.search(pattern, query.lower()):
            return True
    return False

def basic_hallucination_check(response: str, context_docs: List[Dict[str, Any]]) -> float:
    """
    Computes a simplified validation index matching response keywords back to the context.
    Returns a score between 0.0 (High Hallucination) and 1.0 (Completely grounded).
    """
    if not context_docs:
        return 1.0 # No context to conflict with
        
    # Gather structural words out of context
    combined_context = " ".join([doc["content"].lower() for doc in context_docs])
    context_words = set(re.findall(r'\b\w{4,}\b', combined_context)) # check words > 3 characters
    
    if not context_words:
        return 1.0
        
    response_words = re.findall(r'\b\w{4,}\b', response.lower())
    if not response_words:
        return 1.0
        
    matched_words = [word for word in response_words if word in context_words]
    return len(matched_words) / len(response_words)

# Tenacity retry strategy for resilient network calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def execute_with_retry(func, *args, **kwargs):
    """Wraps sensitive backend processing steps within an automatic exponential retry loop."""
    return func(*args, **kwargs)