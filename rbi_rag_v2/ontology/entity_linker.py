import re
from typing import List, Dict

# Simple keyword based linking for MVP
DOMAIN_KEYWORDS = {
    "MSME": ["msme", "micro", "small", "medium", "enterprise", "priority sector"],
    "KCC": ["kcc", "kisan credit card", "agriculture", "farmer"],
    "Basel III": ["basel", "capital", "tier 1", "tier 2", "risk weight", "crar"],
    "NPA": ["npa", "non performing", "asset classification", "provisioning", "sma"],
    "PSL": ["psl", "priority sector", "lending target", "anbc", "weaker section", "sub-target"],
    "KYC": ["kyc", "know your customer", "due diligence", "beneficial owner", "sbo", "aml"],
    "UCB": ["ucb", "urban co-operative", "cooperative bank"],
    "Cybersecurity": ["cybersecurity", "cyber", "csite", "information security", "breach", "incident"],
    "Lead Bank": ["lead bank", "slbc", "dlcc", "branch", "financial inclusion", "unbanked"],
    "DRI": ["dri", "differential rate", "interest rate", "weaker section lending"],
    "PMEGP": ["pmegp", "employment generation", "prime minister"],
    "Liquidity": ["liquidity", "lcr", "nsfr", "funding ratio", "cash reserve"],
}

def extract_entities(query: str) -> List[str]:
    query_lower = query.lower()
    entities = []
    
    # Naive extraction based on keywords
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            entities.append(domain)
            
    return entities
