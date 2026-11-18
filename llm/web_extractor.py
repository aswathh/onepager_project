import json
import re
import requests

from llm.ollama_client import MODEL_NAME

from schemas.field_definitions import (
    get_field_definitions
)


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"

REQUEST_TIMEOUT = 300

MAX_CHUNK_CHARACTERS = 2500

NUM_PREDICT = 512


# ============================================================
# LOAD SHARED FIELD DEFINITIONS
# ============================================================

FIELD_DEFINITIONS = get_field_definitions()


# ============================================================
# NORMALIZE FIELD NAME
# ============================================================

def normalize_field_name(name):
    """
    Normalize field names to support minor differences
    in spaces, hyphens and underscores.
    """

    return re.sub(
        r"[\s\-]+",
        "_",
        str(name).strip().lower()
    )


# ============================================================
# GET FIELD DEFINITION
# ============================================================

def get_field_definition(section, field_name):
    """
    Retrieve the complete shared definition of a field:
        display_name, description, what_to_extract,
        what_not_to_extract, answer_format, source_guidance
    """

    section_definitions = FIELD_DEFINITIONS.get(
        section,
        {}
    )

    normalized_name = normalize_field_name(field_name)

    for defined_name, definition in section_definitions.items():

        if normalize_field_name(defined_name) == normalized_name:

            return {
                "display_name": definition.get(
                    "display_name", defined_name
                ),
                "description": definition.get("description", ""),
                "what_to_extract": definition.get(
                    "what_to_extract", ""
                ),
                "what_not_to_extract": definition.get(
                    "what_not_to_extract", ""
                ),
                "answer_format": definition.get("answer_format", ""),
                "source_guidance": definition.get(
                    "source_guidance", ""
                )
            }

    return None


# ============================================================
# JSON RESPONSE FORMAT
# ============================================================

WEB_ANSWER_FORMAT = {
    "type": "object",

    "properties": {
        "answer": {"type": ["string", "null"]},
        "evidence": {"type": ["string", "null"]},
        "source_id": {"type": ["string", "null"]}
    },

    "required": ["answer", "evidence", "source_id"],

    "additionalProperties": False
}


# ============================================================
# ANSWER SAFETY CHECKS
# ============================================================

# Template text that small models sometimes return as an answer
PLACEHOLDER_ANSWERS = {
    "string", "null", "none", "n/a", "na", "unknown",
    "extracted answer", "extracted answer or null",
    "supporting webpage text", "supporting webpage text or null",
    "source id", "source id or null", "not found",
    "not available", "not specified", "not provided"
}

# Fields whose answer is a short summary and may be paraphrased
SUMMARY_FIELDS = {
    "overview", "description", "route_to_market",
    "vertical_market", "deployment_type"
}

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are",
    "was", "were", "has", "have", "its", "their", "into", "using",
    "used", "also", "such", "can", "will", "which", "other", "more"
}


def alnum_text(text):
    """Lowercase text reduced to letters/digits and single spaces."""

    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split()
    )


def is_placeholder(value):

    if not isinstance(value, str):
        return False

    cleaned = value.strip().lower().strip(".")

    return cleaned in PLACEHOLDER_ANSWERS or " or null" in cleaned


def evidence_in_text(evidence, text):
    """
    Check that the quoted evidence really comes from the text.

    Punctuation, markdown and spacing are ignored. As a fallback the
    evidence passes when 85% of its words occur in the source.
    """

    evidence_norm = alnum_text(evidence)
    text_norm = alnum_text(text)

    if not evidence_norm or not text_norm:
        return False

    if evidence_norm in text_norm:
        return True

    words = evidence_norm.split()

    if len(words) < 3:
        return False

    text_words = set(text_norm.split())

    found = sum(1 for word in words if word in text_words)

    return found / len(words) >= 0.85


def answer_is_grounded(answer, source_text, field_name):
    """
    The answer may only contain facts that exist in the source.

    - every number must appear in the source
    - 60% of the answer words (40% for summary fields) must appear

    This blocks values copied from prompt text or invented from the
    model's own knowledge.
    """

    source_norm = alnum_text(source_text)
    source_compact = source_norm.replace(" ", "")
    source_words = set(source_norm.split())

    for number in re.findall(r"\d[\d,.]*", str(answer)):

        digits = re.sub(r"[^0-9]", "", number)

        if digits and digits not in source_compact:
            return False

    words = [
        word for word in alnum_text(answer).split()
        if len(word) >= 3 and word not in STOPWORDS
    ]

    if not words:
        return True

    found = sum(1 for word in words if word in source_words)

    threshold = 0.4 if field_name in SUMMARY_FIELDS else 0.6

    return found / len(words) >= threshold


def parse_source_id(value):
    """Accept 'S1', 's1', '[SOURCE S1]', 'Source S1' ..."""

    match = re.search(r"s\s*(\d+)", str(value or ""), re.IGNORECASE)

    if not match:
        return None

    return f"S{int(match.group(1))}"


def choose_context_size(prompt_characters, num_predict):
    """
    Pick num_ctx from the real prompt length. A fixed 8192 window can
    silently cut the start of a long prompt (the instructions).
    """

    needed = prompt_characters // 3 + num_predict + 512

    for size in (8192, 16384, 32768):
        if needed <= size:
            return size

    return 32768


def result_without_answer(status):

    return {
        "answer": None,
        "evidence": None,
        "source_url": None,
        "status": status
    }


# ============================================================
# PREPARE EVIDENCE FOR ONE FIELD
# ============================================================

def prepare_field_evidence(evidence_chunks):
    """
    Prepare webpage evidence and assign source IDs.
    """

    sources = []

    evidence_text = []

    for index, chunk in enumerate(evidence_chunks, start=1):

        source_id = f"S{index}"

        source = {
            "source_id": source_id,
            "title": chunk.get("title", ""),
            "url": chunk.get("url", ""),
            "domain": chunk.get("domain", "")
        }

        sources.append(source)

        text = chunk.get("text", "")[:MAX_CHUNK_CHARACTERS]

        evidence_text.append(
            f"""
[SOURCE {source_id}]
Title: {source['title']}
URL: {source['url']}
Domain: {source['domain']}
Content:
{text}
"""
        )

    return "\n".join(evidence_text), sources


# ============================================================
# EXTRACT ONE FIELD USING QWEN
# ============================================================

SYSTEM_PROMPT = """
You are a factual information extraction assistant.

You extract ONE company or product field using ONLY the supplied
webpage evidence.

STRICT RULES:

1. Use only the supplied webpage evidence.
2. Follow the field definition: what to extract, what NOT to
   extract and the expected answer format.
3. The field definition explains what the field means. It is NOT
   data. Never copy words from it into the answer unless the same
   words appear in the webpage evidence.
4. Extract facts only about the TARGET COMPANY / PRODUCT named in
   the request. If the page talks about another company (for
   example a similarly named company or a competitor), ignore it.
5. Do not use outside knowledge. Never guess, assume or invent.
6. Do not confuse similar fields.
7. Copy names, numbers, currencies, units, model names and
   version numbers exactly as written.
8. Do not convert ranges into exact numbers. Do not add missing
   currencies, dates or units.
9. Keep current technologies separate from planned or target ones.
10. Ignore navigation links, unrelated numbers, page metadata and
    profile counts.
11. "evidence" must be a short exact excerpt copied from the
    webpage content that directly supports the answer.
12. "source_id" must be the ID of the source containing that
    excerpt. Never invent a source ID.
13. If the evidence does not clearly answer the field, return null
    for answer, evidence and source_id.
14. Return valid JSON only.

A keyword on a page does not mean it answers the field. Check the
context.
"""


def extract_single_field(
    field_name,
    section,
    field_definition,
    evidence_chunks,
    company_name=None,
    product_name=None
):
    """
    Extract one field using the shared field definition and the
    supplied webpage evidence.

    Every answer is verified: the evidence must exist in a supplied
    page and the answer must be supported by that page.
    """

    if not evidence_chunks:

        return result_without_answer("no_relevant_evidence")

    evidence_text, sources = prepare_field_evidence(
        evidence_chunks
    )

    source_ids = {
        source["source_id"]: source
        for source in sources
    }

    target = ""

    if company_name:
        target += f"TARGET COMPANY: {company_name}\n"

    if product_name:
        target += f"TARGET PRODUCT: {product_name}\n"

    user_prompt = f"""
Extract the requested one-pager field.

{target}
SECTION: {section}
INTERNAL FIELD NAME: {field_name}
DISPLAY NAME: {field_definition.get("display_name", field_name)}

DESCRIPTION:
{field_definition.get("description", "")}

WHAT TO EXTRACT:
{field_definition.get("what_to_extract", "")}

WHAT NOT TO EXTRACT:
{field_definition.get("what_not_to_extract", "")}

EXPECTED ANSWER FORMAT:
{field_definition.get("answer_format", "")}

PREFERRED SOURCES:
{field_definition.get("source_guidance", "")}

AVAILABLE SOURCE IDs: {", ".join(source_ids.keys())}

WEBPAGE EVIDENCE:
{evidence_text}

Return a JSON object with the keys "answer", "evidence" and
"source_id". Use null for all three when the evidence does not
answer this field for the target company.
"""

    # --------------------------------------------------------
    # OLLAMA REQUEST (one retry if the output was cut off)
    # --------------------------------------------------------

    num_predict = NUM_PREDICT

    prompt_characters = len(SYSTEM_PROMPT) + len(user_prompt)

    result = None

    for attempt in range(2):

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "format": WEB_ANSWER_FORMAT,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_ctx": choose_context_size(
                    prompt_characters, num_predict
                ),
                "num_predict": num_predict
            }
        }

        try:

            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            response_data = response.json()

            if (
                response_data.get("done_reason") == "length"
                and attempt == 0
            ):
                num_predict = num_predict * 2
                continue

            result = json.loads(response_data["message"]["content"])

            break

        except requests.exceptions.ConnectionError:

            raise RuntimeError(
                "Could not connect to Ollama. "
                "Check whether Ollama is running."
            )

        except requests.exceptions.Timeout:

            # One slow field must not stop the whole run.
            print(f"Ollama timed out while extracting {field_name}.")

            return result_without_answer("timeout")

        except requests.exceptions.HTTPError as error:

            print(f"Ollama error while extracting {field_name}: {error}")

            return result_without_answer("ollama_error")

        except (ValueError, KeyError, TypeError, IndexError) as error:

            print(
                f"Invalid response while extracting {field_name}: "
                f"{error}"
            )

            return result_without_answer("invalid_model_response")

    if not isinstance(result, dict):

        return result_without_answer("invalid_model_response")

    answer = result.get("answer")
    evidence = result.get("evidence")
    source_id = parse_source_id(result.get("source_id"))

    # ----------------------------------------------------
    # VALIDATE ANSWER
    # ----------------------------------------------------

    if isinstance(answer, list):
        answer = ", ".join(str(item) for item in answer if item)

    if (
        not isinstance(answer, str)
        or not answer.strip()
        or is_placeholder(answer)
    ):

        return result_without_answer("no_supported_answer")

    # ----------------------------------------------------
    # VALIDATE EVIDENCE
    # ----------------------------------------------------

    if (
        not isinstance(evidence, str)
        or not evidence.strip()
        or is_placeholder(evidence)
    ):

        return result_without_answer("missing_evidence")

    # ----------------------------------------------------
    # FIND THE SOURCE OF THE EVIDENCE
    #
    # The model's source_id is tried first. If it is wrong, the
    # evidence is searched in every supplied chunk, so a correct
    # answer is not lost because of a bad ID.
    # ----------------------------------------------------

    candidate_indexes = []

    if source_id in source_ids:
       candidate_indexes.append(int(source_id[1:]) - 1)

    candidate_indexes.extend(
        index for index in range(len(evidence_chunks))
        if index not in candidate_indexes
    )

    selected_index = None

    for index in candidate_indexes:

        chunk_text = evidence_chunks[index].get(
            "text", ""
        )[:MAX_CHUNK_CHARACTERS]

        if evidence_in_text(evidence, chunk_text):
            selected_index = index
            break

    if selected_index is None:

        return result_without_answer("evidence_not_verified")

    selected_chunk_text = evidence_chunks[selected_index].get(
        "text", ""
    )[:MAX_CHUNK_CHARACTERS]

    selected_source = sources[selected_index]

    # ----------------------------------------------------
    # CHECK THAT THE ANSWER IS SUPPORTED BY THE PAGE
    # ----------------------------------------------------

    if not answer_is_grounded(
        answer,
        selected_chunk_text,
        field_name
    ):

        return result_without_answer("answer_not_grounded")

    # ----------------------------------------------------
    # RETURN VERIFIED EXTRACTION
    # ----------------------------------------------------

    return {
        "answer": answer.strip(),
        "evidence": evidence.strip(),
        "source_url": selected_source["url"],
        "source_title": selected_source["title"],
        "status": "extracted"
    }


# ============================================================
# EXTRACT ALL MISSING FIELDS
# ============================================================

def extract_from_web(
    filtered_evidence,
    company_name=None,
    product_name=None
):
    """
    Extract all missing fields using shared field definitions.

    Each field is processed separately. The result is the input of
    merge_extractions(), which puts every answer with status
    "extracted" into the matching empty field.

    company_name / product_name are optional but recommended: they
    tell the model whose facts to extract.
    """

    web_results = {}

    total_fields = len(filtered_evidence)

    print("\n========== QWEN WEB EXTRACTION ==========")

    for index, (field_name, field_data) in enumerate(
        filtered_evidence.items(),
        start=1
    ):

        section = field_data.get("section", "")

        evidence_chunks = field_data.get("evidence", [])

        print(
            f"\n[{index}/{total_fields}] "
            f"Extracting: {field_name}"
        )

        print("Section:", section)

        print("Relevant chunks:", len(evidence_chunks))

        definition = get_field_definition(
            section=section,
            field_name=field_name
        )

        if definition is None:

            print("Status: field_definition_not_found")

            web_results[field_name] = {
                "section": section,
                "answer": None,
                "evidence": None,
                "source_url": None,
                "status": "field_definition_not_found"
            }

            continue

        result = extract_single_field(
            field_name=field_name,
            section=section,
            field_definition=definition,
            evidence_chunks=evidence_chunks,
            company_name=company_name,
            product_name=product_name
        )

        web_results[field_name] = {
            "section": section,
            **result
        }

        print("Status:", result["status"])

        print("Answer:", result["answer"])

        if result.get("source_url"):

            print("Source:", result["source_url"])

        if result.get("evidence"):

            print("Evidence:", result["evidence"])

    print("\n========== WEB EXTRACTION COMPLETED ==========")

    print("Total fields processed:", total_fields)

    successful = sum(
        1
        for result in web_results.values()
        if result.get("status") == "extracted"
    )

    print("Successfully extracted:", successful)

    print(
        "Not extracted:",
        total_fields - successful
    )

    return web_results