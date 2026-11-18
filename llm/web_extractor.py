 
import json
import re
import requests

from llm.ollama_client import MODEL_NAME

from schemas.field_definitions import (
    get_field_definitions,
    build_extraction_instruction
)


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"

REQUEST_TIMEOUT = 300

MAX_CHUNK_CHARACTERS = 2500


# ============================================================
# LOAD SHARED FIELD DEFINITIONS
# ============================================================

FIELD_DEFINITIONS = get_field_definitions()


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


def get_field_definition(section, field_name):
    """
    Retrieve the description and extraction instructions
    from the shared field_definitions.py file.
    """

    section_definitions = FIELD_DEFINITIONS.get(
        section,
        {}
    )

    normalized_name = normalize_field_name(field_name)

    for defined_name, definition in section_definitions.items():

        if normalize_field_name(defined_name) == normalized_name:
            return definition

    return None


# ============================================================
# JSON RESPONSE FORMAT
# ============================================================

WEB_ANSWER_FORMAT = {
    "type": "object",
    "properties": {
        "answer": {
            "type": ["string", "null"]
        },
        "evidence": {
            "type": ["string", "null"]
        },
        "source_id": {
            "type": ["string", "null"]
        }
    },
    "required": [
        "answer",
        "evidence",
        "source_id"
    ],
    "additionalProperties": False
}


# ============================================================
# ANSWER SAFETY CHECKS
# ============================================================

# Small models sometimes return the template text of the prompt
# instead of an answer.
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
    """Lowercase text reduced to letters/digits separated by spaces."""

    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split()
    )


def is_placeholder(value):
    """Detect template text returned as if it were an answer."""

    if not isinstance(value, str):
        return False

    cleaned = value.strip().lower().strip(".")

    if cleaned in PLACEHOLDER_ANSWERS:
        return True

    return " or null" in cleaned


def evidence_in_text(evidence, text):
    """
    Check that the quoted evidence really comes from the text.

    Exact substring matching failed on harmless differences
    (markdown links, bullets, quotes, double spaces), so punctuation
    is ignored. As a fallback, the evidence passes when at least
    85% of its words occur in the source.
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
    Check that the answer only contains facts present in the source.

    - every number in the answer must appear in the source
    - enough of the answer words must appear in the source
      (60% for list/fact fields, 40% for summary fields)

    This blocks values copied from prompt examples or invented
    from the model's own knowledge.
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

        text = chunk.get("text", "")

        text = text[:MAX_CHUNK_CHARACTERS]

        evidence_text.append(
            f"""
[SOURCE {source_id}]
Title: {source['title']}
URL: {source['url']}
Content:
{text}
"""
        )

    return "\n".join(evidence_text), sources


# ============================================================
# EXTRACT ONE FIELD USING QWEN
# ============================================================

def choose_context_size(prompt_characters, num_predict):
    """
    Pick num_ctx so the prompt is never silently truncated.

    Ollama cuts the beginning of an over-long prompt without warning,
    which removes the instructions and evidence.
    """

    needed = prompt_characters // 3 + num_predict + 512

    for size in (4096, 8192, 16384, 32768):
        if needed <= size:
            return max(size, 8192)

    return 32768


def result_without_answer(status):

    return {
        "answer": None,
        "evidence": None,
        "source_url": None,
        "status": status
    }


def extract_single_field(
    field_name,
    section,
    description,
    extraction_instruction,
    evidence_chunks
):
    """
    Extract one field using the shared field definition
    and supplied webpage evidence.
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

    # --------------------------------------------------------
    # SYSTEM PROMPT  (rules only - no example answers)
    # --------------------------------------------------------

    system_prompt = """
You are a factual information extraction assistant.

You extract one company or product field using ONLY the
supplied webpage evidence.

RULES:

1. Use only the supplied webpage evidence.
2. Do not use outside knowledge or previously learned facts.
3. Never guess, assume or invent information.
4. The field description and instructions explain what the field
   means. They are NOT data. Never copy words from them into the
   answer unless the same words appear in the evidence.
5. Copy names, numbers, currencies and units exactly as written
   in the evidence.
6. Do not convert ranges into exact values.
7. If the evidence does not clearly answer the field, return null
   for answer, evidence and source_id.
8. "evidence" must be a short exact excerpt copied from the
   webpage content that directly supports the answer.
9. "source_id" must be one of the supplied source IDs.
10. Return valid JSON only.

A keyword appearing on a page does not mean it answers the field.
Check the context. Ignore navigation links, profile counts and
page metadata.
"""

    # --------------------------------------------------------
    # USER PROMPT
    # --------------------------------------------------------

    user_prompt = f"""
SECTION: {section}
FIELD NAME: {field_name}

FIELD DESCRIPTION:
{description}

FIELD INSTRUCTIONS:
{extraction_instruction}

AVAILABLE SOURCE IDs: {", ".join(source_ids.keys())}

WEBPAGE EVIDENCE:
{evidence_text}

Return a JSON object with the keys "answer", "evidence" and
"source_id". Use null for all three when the evidence does not
answer the field.
"""

    # --------------------------------------------------------
    # OLLAMA REQUEST (one retry if the output was cut off)
    # --------------------------------------------------------

    num_predict = 512
    prompt_characters = len(system_prompt) + len(user_prompt)

    result = None

    for attempt in range(2):

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "format": WEB_ANSWER_FORMAT,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_ctx": choose_context_size(
                    prompt_characters,
                    num_predict
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
                num_predict = 1024
                continue

            content = response_data["message"]["content"]

            result = json.loads(content)

            break

        except requests.exceptions.ConnectionError:

            raise RuntimeError(
                "Could not connect to Ollama. "
                "Check whether Ollama is running."
            )

        except requests.exceptions.Timeout:

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

    # --------------------------------------------------------
    # VALIDATE ANSWER
    # --------------------------------------------------------

    if (
        not isinstance(answer, str)
        or not answer.strip()
        or is_placeholder(answer)
    ):

        return result_without_answer("no_supported_answer")

    # --------------------------------------------------------
    # VALIDATE EVIDENCE
    # --------------------------------------------------------

    if (
        not isinstance(evidence, str)
        or not evidence.strip()
        or is_placeholder(evidence)
    ):

        return result_without_answer("missing_evidence")

    # --------------------------------------------------------
    # LOCATE THE SOURCE OF THE EVIDENCE
    #
    # The model's source_id is tried first. If it is wrong or
    # missing, the evidence is searched in every supplied chunk so a
    # correct answer is not lost because of a bad ID.
    # --------------------------------------------------------

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

    selected_chunk = evidence_chunks[selected_index]
    selected_source = sources[selected_index]

    # --------------------------------------------------------
    # CHECK THAT THE ANSWER IS SUPPORTED BY THE SOURCE
    # --------------------------------------------------------

    if not answer_is_grounded(
        answer,
        selected_chunk.get("text", ""),
        field_name
    ):

        return result_without_answer("answer_not_grounded")

    # --------------------------------------------------------
    # RETURN VERIFIED EXTRACTION
    # --------------------------------------------------------

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

def extract_from_web(filtered_evidence):
    """
    Extract all missing fields using shared field definitions.

    Qwen receives the appropriate description and extraction
    instructions for every field.
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

        # ----------------------------------------------------
        # GET SHARED FIELD DEFINITION
        # ----------------------------------------------------

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

        description = definition.get("description", "")

        extraction_instruction = build_extraction_instruction(
            definition
        )

        # ----------------------------------------------------
        # CALL QWEN
        # ----------------------------------------------------

        try:

            result = extract_single_field(
                field_name=field_name,
                section=section,
                description=description,
                extraction_instruction=extraction_instruction,
                evidence_chunks=evidence_chunks
            )

        except RuntimeError:
            raise

        web_results[field_name] = {
            "section": section,
            **result
        }

        print("Status:", result["status"])

        print("Answer:", result["answer"])

        if result.get("source_url"):

            print("Source:", result["source_url"])

    return web_results
