import json
import logging
from generation.groq_client import generate_groq_completion

ACRONYMS = {
    "DAY-NRLM": "Deendayal Antyodaya Yojana - National Rural Livelihoods Mission",
    "PMJDY": "Pradhan Mantri Jan-Dhan Yojana",
    "DRI": "Differential Rate of Interest",
    "KCC": "Kisan Credit Card",
    "MSME": "Micro, Small and Medium Enterprises",
    "PMEGP": "Prime Minister's Employment Generation Programme",
    "RRB": "Regional Rural Bank",
    "UCB": "Urban Co-operative Bank",
    "PSL": "Priority Sector Lending",
    "CRAR": "Capital to Risk (Weighted) Assets Ratio",
    "AT1": "Additional Tier 1",
    "PDI": "Perpetual Debt Instruments",
    "SMF": "Small and Marginal Farmers",
    "ANBC": "Adjusted Net Bank Credit",
    "CEOBSE": "Credit Equivalent Amount of Off-Balance Sheet Exposure"
}

def expand_query(query: str) -> str:
    import re
    expanded = query
    for acronym, full_form in ACRONYMS.items():
        # Case insensitive exact word match
        pattern = re.compile(r'\b' + re.escape(acronym) + r'\b', re.IGNORECASE)
        expanded = pattern.sub(f"{acronym} ({full_form})", expanded)
    return expanded

class QueryClassifier:
    def __call__(self, query: str) -> str:
        prompt = f"""Classify the following query into one of three categories:
1. OUT_OF_CONTEXT: ONLY if the query is completely unrelated to banking, finance, RBI regulations, or Indian financial sector policy. Examples: questions about cooking, movies, geography, medicine, sports.
2. COMPOUND: If it contains multiple sub-questions or requires fetching from distinct regulatory areas.
3. SIMPLE: If it is a single regulatory lookup. This includes ANY query about: priority sector lending (PSL), Basel III, KCC, MSME, DAY-NRLM, DRI scheme, microfinance, bank guarantees, capital adequacy, credit flow, SHGs, interest subvention, lead bank scheme, PMJDY, UCBs, RRBs, PDIs, ANBC, CGTMSE, NBFCs, or any other RBI/banking/finance topic.

IMPORTANT: If the query mentions ANY banking/finance/RBI acronym or concept, classify as SIMPLE or COMPOUND — never OUT_OF_CONTEXT.

Query: {query}
Output ONLY the category name (OUT_OF_CONTEXT, COMPOUND, or SIMPLE)."""
        try:
            res = generate_groq_completion(prompt, stream=False).strip()
            if "OUT_OF_CONTEXT" in res: return "OUT_OF_CONTEXT"
            if "COMPOUND" in res: return "COMPOUND"
            return "SIMPLE"
        except Exception as e:
            logging.error(f"Classification failed: {e}")
            return "SIMPLE"

class QueryDecomposer:
    def __call__(self, query: str) -> list:
        prompt = f"""Decompose this regulatory query into 2-4 simpler, standalone sub-queries.
Return ONLY a valid JSON array of strings.
Query: {query}"""
        try:
            res = generate_groq_completion(prompt, stream=False)
            import re
            json_str = re.sub(r'```(?:json)?\n?(.*?)\n?```', r'\1', res, flags=re.DOTALL).strip()
            sub_queries = json.loads(json_str)
            if isinstance(sub_queries, list):
                return sub_queries
        except Exception as e:
            logging.error(f"Decomposition failed: {e}")
        return [query]

def decompose_query(query: str) -> dict:
    """Legacy endpoint wrapper."""
    decomposer = QueryDecomposer()
    sub_queries = decomposer(query=query)
    return {"sub_queries": sub_queries, "domain": None}

def parallel_subquery_retrieve(sub_queries: list, original_query: str = None) -> list:
    from retrieval.fusion import hybrid_search
    # Naive parallel/union retrieve
    all_results = []
    seen = set()
    for sq in sub_queries:
        res = hybrid_search(sq)
        for r in res:
            if r["id"] not in seen:
                seen.add(r["id"])
                all_results.append(r)
    # Re-rank across unioned set
    if not all_results:
        return []
    from retrieval.fusion import rerank_unioned_results
    rerank_query = original_query or " ".join(sub_queries)
    return rerank_unioned_results(rerank_query, all_results)

