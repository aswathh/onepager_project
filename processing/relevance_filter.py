 
import re


# ============================================================
# Field-specific relevance keywords
# ============================================================

FIELD_KEYWORDS = {

    "employee_count": [
        "employee count", "number of employees",
        "employees worldwide", "company size",
        "headcount", "workforce", "team size",
        "employees", "staff"
    ],

    "headquarters": [
        "headquarters", "headquartered",
        "based in", "head office",
        "corporate office", "registered office",
        "headquarter",
        "head quarters",
        "our office",
        "main office",
        "located in",
        "address"
    ],

    "funding": [
        "total funding", "funding amount",
        "raised funding", "total raised",
        "investment", "funding round",
        "series a", "series b", "venture capital",
        "funding",
        "raised",
        "seed round",
        "pre-seed",
        "series c",
        "investors",
        "backed by"
    ],

    "revenue": [
        "annual revenue", "company revenue",
        "reported revenue", "total revenue",
        "sales revenue", "turnover",
        "revenue",
        "arr",
        "annual recurring revenue",
        "net sales"
    ],

    "gtm_partners": [
        "official partner", "channel partner",
        "distribution partner", "reseller partner",
        "strategic partnership", "implementation partner",
        "technology partner", "distribution agreement",
        "partner network", "partner ecosystem",
        "partners",
        "partner",
        "partnership",
        "partnered with",
        "resellers",
        "distributors",
        "alliance"
    ],

    "key_customers": [
        "our customers", "our clients",
        "customer case study", "client case study",
        "customer success story", "client success story",
        "deployed at", "used by",
        "trusted by", "customers include",
        "clients include",
        "customers",
        "clients",
        "case study",
        "case studies",
        "testimonials",
        "used by",
        "works with"
    ],

    "competitors": [
        "direct competitors", "main competitors",
        "key competitors", "competitors include",
        "competes with", "competing against",
        "alternatives to", "alternative to",
        "compared with", "competitive landscape",
        "competitors",
        "competitor",
        "alternatives",
        "versus",
        "vs",
        "competition"
    ],

    "route_to_market": [
        "direct sales", "sales channel",
        "distribution channel", "reseller",
        "subscription model", "marketplace",
        "channel sales", "go-to-market strategy",
        "sales model", "commercial model",
        "pricing",
        "plans",
        "partner program",
        "direct",
        "saas",
        "book a demo",
        "request a demo"
    ],

    "inference_hardware": [
        "inference hardware", "inference server",
        "inference accelerator", "gpu inference",
        "edge inference", "nvidia gpu",
        "intel accelerator",
        "inference",
        "gpu",
        "gpus",
        "cpu",
        "cpus",
        "accelerator",
        "accelerators"
    ],

    "deployment_type": [
        "cloud deployment", "cloud-based",
        "on-premises", "on premise",
        "edge deployment", "hybrid deployment",
        "deployment model", "self-hosted"
    ],

    "training_hardware": [
        "training hardware", "model training",
        "training gpu", "gpu cluster",
        "training infrastructure", "nvidia a100",
        "nvidia h100", "google tpu",
        "training",
        "gpu",
        "gpus",
        "tpu",
        "tpus"
    ],

    "models": [
        "machine learning model", "language model",
        "neural network", "transformer model",
        "bert model", "large language model",
        "model architecture", "foundation model",
        "model",
        "models",
        "llm",
        "llms",
        "bert",
        "gpt",
        "architecture"
    ],

    "ai_framework": [
        "pytorch", "tensorflow",
        "hugging face transformers", "peft",
        "lora", "keras", "training framework"
    ],

    "serving_framework": [
        "model serving", "inference runtime",
        "onnx runtime", "tensorrt",
        "vllm", "torchserve", "triton inference"
    ],

    "additional_ai_software_stack": [
        "vector database", "graph database",
        "mlops", "data pipeline",
        "model monitoring", "observability",
        "feature store", "kafka",
        "database",
        "postgres",
        "redis",
        "elasticsearch",
        "airflow",
        "docker",
        "kubernetes"
    ],

    "vertical_market": [
        "healthcare industry", "education industry",
        "financial services", "manufacturing industry",
        "vertical market", "industries served",
        "industry use cases", "target industries",
        "industries",
        "industry",
        "healthcare",
        "education",
        "finance",
        "retail",
        "manufacturing",
        "sectors"
    ],

    "overview": [
        "about us", "who we are", "what we do",
        "our mission", "company overview", "our story",
        "we help", "we provide", "we build", "founded"
    ],

    "description": [
        "product overview", "platform overview",
        "product features", "platform capabilities",
        "product capabilities", "what we do",
        "technology platform", "product description"
    ]
}


# ============================================================
# Field-specific minimum relevance scores
# ============================================================

# One strong keyword (+2), one phrase (+3) or one fact-pattern (+3)
# is enough. The previous thresholds (2-3) rejected a chunk such as
# "Headquarters: <city>" or "25 employees" because it only contained
# one single-word keyword, so real evidence never reached the model.
DEFAULT_MINIMUM_SCORE = 2

FIELD_MINIMUM_SCORES = {}


# Single words that are strong evidence on their own (+2 instead of +1)
STRONG_WORDS = {
    "employees", "headcount", "headquarters", "headquartered",
    "revenue", "turnover", "funding", "raised",
    "customers", "clients", "partners", "resellers", "distributors",
    "competitors", "alternatives", "pricing", "industries",
    "inference", "training", "overview"
}


# Fact-shaped patterns. A match is worth +3 and ranks real facts
# above chunks that merely contain a keyword in a menu or footer.
FIELD_PATTERNS = {
    "employee_count": [
        r"\b\d[\d,]*\+?\s*(?:-|to)\s*\d[\d,]*\s+employees",
        r"\b\d[\d,]*\+?\s+employees",
        r"employees\s*[:\-]\s*\d",
        r"company size\s*[:\-]?\s*\d"
    ],
    "headquarters": [
        r"headquarter(?:s|ed)?\s*(?:in|at|:|-)",
        r"based (?:in|out of)\s+[a-z]"
    ],
    "funding": [
        r"(?:raised|funding|investment)[^.\n]{0,40}(?:\$|€|£|₹|usd|eur|inr)\s?\d",
        r"(?:\$|€|£|₹)\s?\d[\d,.]*\s?(?:k|m|mn|million|b|bn|billion)\b[^.\n]{0,40}(?:round|funding|raised|investment)"
    ],
    "revenue": [
        r"revenue[^.\n]{0,40}(?:\$|€|£|₹|usd|eur|inr)\s?\d",
        r"(?:\$|€|£|₹)\s?\d[\d,.]*\s?(?:k|m|mn|million|b|bn|billion)\b[^.\n]{0,30}revenue"
    ]
}


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text):
    """Normalize text for keyword matching."""

    text = (text or "").lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# Identity verification check
# ============================================================

def is_identity_verified(chunk):
    """
    Check whether the chunk passed company identity verification.

    Supports identity metadata at the top level or
    inside a metadata dictionary.
    """

    if chunk.get("identity_verified") is True:
        return True

    metadata = chunk.get("metadata", {})

    if isinstance(metadata, dict):

        if metadata.get("identity_verified") is True:
            return True

    return False


# ============================================================
# Relevance score calculation
# ============================================================

def calculate_relevance_score(text, field):
    """
    Calculate field-specific relevance.

    Phrase keyword   +3
    Strong word      +2
    Normal word      +1
    Fact pattern     +3
    """

    text_lower = normalize_text(text)

    keywords = FIELD_KEYWORDS.get(
        field,
        [field.replace("_", " ")]
    )

    score = 0
    matched_keywords = []

    for keyword in keywords:

        keyword = normalize_text(keyword)

        if not keyword:
            continue

        # Phrase matching
        if " " in keyword or "-" in keyword:

            if keyword in text_lower:

                score += 3
                matched_keywords.append(keyword)

        # Individual word matching
        else:

            pattern = (
                r"\b"
                + re.escape(keyword)
                + r"\b"
            )

            if re.search(pattern, text_lower):

                score += 2 if keyword in STRONG_WORDS else 1
                matched_keywords.append(keyword)

    for pattern in FIELD_PATTERNS.get(field, []):

        if re.search(pattern, text_lower):

            score += 3
            matched_keywords.append("pattern:" + pattern[:25])

    return {
        "score": score,
        "matched_keywords": matched_keywords
    }


# ============================================================
# Filter chunks for one field
# ============================================================

MAX_CHUNKS_PER_URL = 2

# For these fields the extractor tries the official website first,
# then LinkedIn, then other sites. So the selection must contain
# chunks of every source type, not only the highest-scoring ones
# (which may all come from third-party pages).
SOURCE_QUOTA_FIELDS = {"headquarters", "employee_count"}

SOURCE_QUOTAS = {
    "official_website": 2,
    "linkedin": 1,
    "other": 1
}


def filter_chunks_for_field(
    chunks,
    field,
    top_k=3,
    minimum_score=None
):
    """
    Select relevant and identity-verified chunks
    for one missing field.
    """

    scored_chunks = []

    if minimum_score is None:

        minimum_score = FIELD_MINIMUM_SCORES.get(
            field,
            DEFAULT_MINIMUM_SCORE
        )

    for chunk in chunks:

        # ----------------------------------------------------
        # Identity verification is mandatory
        # ----------------------------------------------------

        if not is_identity_verified(chunk):
            continue

        # ----------------------------------------------------
        # Relevance calculation
        # ----------------------------------------------------

        relevance = calculate_relevance_score(
            chunk.get("text", ""),
            field
        )

        if relevance["score"] < minimum_score:
            continue

        score = relevance["score"]

        # A page that was found by searching for this field gets a
        # bonus, but it is no longer a hard requirement: the
        # official homepage found for "headquarters" also contains
        # the employee count, customers and partners.
        related_fields = chunk.get("related_fields", [])

        if related_fields and field in related_fields:
            score += 2

        scored_chunks.append({
            **chunk,
            "relevance_score": score,
            "matched_keywords": relevance["matched_keywords"]
        })

    # Highest relevance first, official sources first on ties
    scored_chunks.sort(
        key=lambda item: (
            -item["relevance_score"],
            item.get("source_priority", 4)
        )
    )

    # Do not let one long page fill every slot
    selected = []
    per_url = {}
    per_source = {}

    def take(chunk):

        url = chunk.get("url", "")

        if per_url.get(url, 0) >= MAX_CHUNKS_PER_URL:
            return False

        per_url[url] = per_url.get(url, 0) + 1

        selected.append(chunk)

        return True

    if field in SOURCE_QUOTA_FIELDS:

        limit = max(top_k, sum(SOURCE_QUOTAS.values()))

        # Pass 1: guarantee chunks of every source type
        # (official website, LinkedIn, other), best scores first.
        for chunk in scored_chunks:

            source = chunk.get("source_type", "other")

            if source not in SOURCE_QUOTAS:
                source = "other"

            if per_source.get(source, 0) >= SOURCE_QUOTAS[source]:
                continue

            if take(chunk):
                per_source[source] = per_source.get(source, 0) + 1

            if len(selected) >= limit:
                break

        # Pass 2: fill the remaining slots by relevance, so a field
        # that only has third-party pages still gets top_k chunks.
        for chunk in scored_chunks:

            if len(selected) >= top_k:
                break

            if chunk in selected:
                continue

            take(chunk)

        return selected

    for chunk in scored_chunks:

        take(chunk)

        if len(selected) >= top_k:
            break

    return selected


# ============================================================
# Filter chunks for all missing fields
# ============================================================

def filter_chunks_for_missing_fields(
    chunks,
    missing_report,
    top_k=3,
    minimum_score=None
):
    """
    Filter crawled chunks for every missing field.

    Only identity-verified chunks are considered.

    Returns evidence grouped by company/product field.
    """

    filtered_results = {}

    missing_fields = missing_report.get(
        "all_missing_fields",
        []
    )

    print("\n========== EVIDENCE RELEVANCE FILTER ==========")

    for index, field_info in enumerate(
        missing_fields,
        start=1
    ):

        section = field_info["section"]
        field = field_info["field"]

        print(
            f"\n[{index}/{len(missing_fields)}] "
            f"Filtering: {field}"
        )

        relevant_chunks = filter_chunks_for_field(
            chunks=chunks,
            field=field,
            top_k=top_k,
            minimum_score=minimum_score
        )

        filtered_results[field] = {
            "section": section,
            "description": field_info.get(
                "description", ""
            ),
            "chunks_found": len(relevant_chunks),
            "evidence": relevant_chunks
        }

        print(
            "Relevant verified chunks:",
            len(relevant_chunks)
        )

        if not relevant_chunks:
            print("No suitable evidence found.")

    return filtered_results
