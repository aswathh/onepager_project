 
import json
import re
import requests

from schemas.one_pager_schema import OnePagerExtraction
from schemas.field_definitions import get_field_definitions
from llm.ollama_client import MODEL_NAME


OLLAMA_URL = "http://localhost:11434/api/chat"
REQUEST_TIMEOUT = 300


# ============================================================
# FIELD DEFINITIONS WITHOUT EXAMPLES
# ============================================================

NAME_FIELDS = {"company_name", "product_name"}


def get_extraction_definitions():
    """
    Prepare field definitions for the questionnaire prompt.

    Includes:
    - Description
    - What to extract
    - What not to extract
    - Answer format

    Left out on purpose:
    - company_name / product_name (supplied by the user)
    - source_guidance (describes web sources; useless for a
      questionnaire and wasted about a quarter of the prompt)

    Examples are intentionally excluded.
    """

    definitions = get_field_definitions()

    clean_definitions = {}

    for section, fields in definitions.items():

        clean_definitions[section] = {}

        for field_name, definition in fields.items():

            if field_name in NAME_FIELDS:
                continue

            clean_definitions[section][field_name] = {
                "description": definition.get("description", ""),
                "what_to_extract": definition.get("what_to_extract", ""),
                "what_not_to_extract": definition.get("what_not_to_extract", ""),
                "answer_format": definition.get("answer_format", "")
            }

    return clean_definitions


def get_ollama_schema():
    """
    JSON schema sent to Ollama WITHOUT descriptions/titles.

    The structured-output schema is part of the prompt. Descriptions
    inside it were copied into answers by the model. The meaning of
    every field is already given by the field definitions.
    """

    schema = OnePagerExtraction.model_json_schema()

    def strip(node):

        if isinstance(node, dict):

            return {
                key: strip(value)
                for key, value in node.items()
                if key not in ("description", "title", "examples")
            }

        if isinstance(node, list):
            return [strip(item) for item in node]

        return node

    return strip(schema)


# ============================================================
# NULL NORMALIZATION
# ============================================================

NULL_STRINGS = {
    "null", "none", "n/a", "na", "not available", "not specified",
    "not provided", "unknown", "string", "not found",
    "extracted answer", "extracted answer or null"
}


def normalize_null_values(data):

    if isinstance(data, dict):

        return {
            key: normalize_null_values(value)
            for key, value in data.items()
        }

    elif isinstance(data, list):

        items = [
            normalize_null_values(item)
            for item in data
        ]

        return [item for item in items if item is not None]

    elif isinstance(data, str):

        cleaned = data.strip().lower().strip(".")

        if cleaned in NULL_STRINGS or " or null" in cleaned:
            return None

    return data


def sanitize_to_schema(data):
    """
    Make the model output fit OnePagerExtraction.

    - drops unknown keys (the schema forbids extras)
    - converts lists to comma-separated text (partners, customers,
      competitors are often returned as lists)
    - converts numbers to text
    """

    clean = {"company": {}, "product": {}}

    definitions = get_field_definitions()

    for section in ("company", "product"):

        section_data = data.get(section, {}) if isinstance(data, dict) else {}

        if not isinstance(section_data, dict):
            continue

        for field_name in definitions[section]:

            value = section_data.get(field_name)

            if isinstance(value, list):

                value = ", ".join(
                    str(item).strip() for item in value
                    if item is not None and str(item).strip()
                ) or None

            elif isinstance(value, (int, float)) and not isinstance(value, bool):

                value = str(value)

            elif isinstance(value, dict):

                value = json.dumps(value, ensure_ascii=False)

            clean[section][field_name] = value

    return clean


# ============================================================
# OLLAMA REQUEST
# ============================================================

def choose_context_size(messages, num_predict):
    """
    Choose num_ctx from the real prompt length.

    A fixed 8192 window was smaller than field definitions +
    questionnaire + schema. Ollama then silently drops part of the
    prompt, the questionnaire is cut off, and the model fills the
    fields from the schema/definition text instead of the document.
    """

    prompt_characters = sum(
        len(message["content"]) for message in messages
    )

    needed = prompt_characters // 3 + num_predict + 512

    for size in (8192, 16384, 32768):
        if needed <= size:
            return size

    print(
        "\nWARNING: prompt is very long "
        f"(~{needed} tokens). Part of it may be ignored."
    )

    return 32768


def call_ollama(messages, response_format, num_predict=4096):

    for attempt in range(2):

        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "format": response_format,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_ctx": choose_context_size(
                    messages,
                    num_predict
                ),
                "num_predict": num_predict
            }
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        response_data = response.json()

        done_reason = response_data.get("done_reason")

        print("\nDone reason:", done_reason)

        # Output cut off -> JSON would be invalid. Retry once
        # with more room.
        if done_reason == "length" and attempt == 0:
            num_predict = num_predict * 2
            continue

        message = response_data.get("message", {})

        raw_output = message.get("content", "")

        if not raw_output or not raw_output.strip():

            raise ValueError(
                "Ollama returned empty message content."
            )

        return raw_output

    raise ValueError("Ollama output was cut off twice.")


# ============================================================
# ORIGINAL QUESTIONNAIRE ANSWER EXTRACTION
# ============================================================

def extract_questionnaire_answers(
    questionnaire_text,
    company_name,
    product_name
):

    field_definitions = get_extraction_definitions()

    schema = get_ollama_schema()

    system_prompt = """
You are a professional company and product information
extraction system.

Your task is to extract accurate information from
the supplied questionnaire.

You must understand the meaning of each one-pager field
and identify the corresponding information in the
questionnaire.

STRICT RULES:

1. Use ONLY the supplied questionnaire.
2. Do not use external knowledge or previously learned facts.
3. Never invent or assume information.
4. Return null only when the information is genuinely
   unavailable or cannot be identified.
5. Understand all properties of each field definition
   before extracting its answer.
6. Do not confuse similar fields.
7. Do not assign information belonging to another company
   to the requested company.
8. Preserve the original meaning of the questionnaire.
9. Preserve numerical values, units and currencies exactly.
10. Include every required field.
11. Follow the supplied JSON schema exactly.
12. Return valid JSON only.

IMPORTANT ABOUT FIELD DEFINITIONS:

Description explains what information is required.

What_to_extract explains which information should be identified.

What_not_to_extract explains which information must be excluded.

Answer_format explains how the answer should be presented.

Extract only information actually supported by the
questionnaire.

These definitions are instructions, NOT source data.

Never use hypothetical values or previously learned
company information as actual answers.

Extract actual values from the questionnaire.
"""

    user_prompt = f"""
REQUESTED COMPANY:

{company_name}


REQUESTED PRODUCT:

{product_name}


FIELD DEFINITIONS:

{json.dumps(
    field_definitions,
    indent=2,
    ensure_ascii=False
)}


QUESTIONNAIRE CONTENT:

{questionnaire_text}


EXTRACTION INSTRUCTIONS:

Extract all company and product fields using the
provided field definitions.

For each field:

1. Understand what the field represents.
2. Read all properties in its definition.
3. Search the questionnaire for relevant information.
4. Identify the correct information.
5. Check that the information does not belong to
   another field or another company.
6. Return the answer in the specified answer format.
7. Return null if the information is not supported
   by the questionnaire.

IMPORTANT FIELD DISTINCTIONS:

COMPANY INFORMATION:

- Overview: What the company does and what it offers.
- Headquarters: The company's actual headquarters.
- Employee count: Number or range of employees.
- Funding: Money raised or received by the company.
- Revenue: Money earned through business operations.
- GTM partners: Organizations helping sell or distribute
  the company's products.
- Key customers: Organizations actually using or buying
  the company's products.
- Competitors: Companies offering competing solutions.

PRODUCT INFORMATION:

- Description: What the product does and what problem
  it solves.
- Vertical market: Industries and application domains.
- Route to market: How the product is commercially
  sold or distributed.
- Inference hardware: Hardware used to execute models.
- Deployment type: Where the product is deployed.
- Training hardware: Hardware used to train models.
- Models: Model architectures and model families.
- AI framework: Frameworks used for model development
  and training.
- Serving framework: Frameworks used for production
  inference and model serving.
- Additional AI software stack: Supporting AI
  infrastructure and software.

SPECIAL INSTRUCTIONS:

1. Preserve numerical values, units and currencies exactly
   as written in the questionnaire.

2. Preserve revenue suffixes and scale indicators.
   Do not remove M, K, B, million, billion, or currency
   information from the original answer.

3. A technology partnership does not automatically
   establish a commercial sales channel.

4. A deployment statement does not automatically
   establish a route-to-market strategy.

5. Extract models, hardware and frameworks independently.
   Do not mix their categories.

6. Do not fill a field merely because a similar value
   appears in another field.

7. Do not assume that a company operating in AI is
   automatically a competitor of another AI company.

8. A company address is not automatically its headquarters.
   Only identify headquarters when the questionnaire
   supports that classification.

9. Do not convert an employee range into an exact
   employee count.

10. Do not treat a LinkedIn profile count as the company's
    declared employee count.

Return the complete JSON object matching the schema.
"""

    print("\n========== QUESTIONNAIRE ANSWER EXTRACTION ==========")

    print("Connecting to Ollama...")
    print("Model:", MODEL_NAME)

    raw_output = call_ollama(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        response_format=schema,
        num_predict=4096
    )

    print("\n========== RAW QUESTIONNAIRE EXTRACTION ==========")

    print(raw_output)

    parsed_data = json.loads(raw_output)

    parsed_data = normalize_null_values(parsed_data)

    parsed_data = sanitize_to_schema(parsed_data)

    extracted_data = OnePagerExtraction.model_validate(
        parsed_data
    )

    # User-provided names are authoritative.
    extracted_data.company.company_name = company_name
    extracted_data.product.product_name = product_name

    return extracted_data


# ============================================================
# EVIDENCE RESPONSE SCHEMA
# ============================================================

def get_evidence_schema():

    definitions = get_field_definitions()

    evidence_item = {
        "type": "object",
        "properties": {
            "evidence": {
                "type": ["string", "null"]
            }
        },
        "required": ["evidence"],
        "additionalProperties": False
    }

    return {
        "type": "object",
        "properties": {
            section: {
                "type": "object",
                "properties": {
                    field: evidence_item
                    for field in fields
                },
                "required": list(fields.keys()),
                "additionalProperties": False
            }
            for section, fields in definitions.items()
        },
        "required": list(definitions.keys()),
        "additionalProperties": False
    }


# ============================================================
# SEPARATE QUESTIONNAIRE EVIDENCE EXTRACTION
# ============================================================

def extract_questionnaire_evidence(
    extracted_data,
    questionnaire_text
):

    answer_data = extracted_data.model_dump()

    schema = get_evidence_schema()

    system_prompt = """
You are a source evidence identification system.

Your task is to identify the original questionnaire
passages supporting already-extracted answers.

Do not generate new answers.

STRICT RULES:

1. Use only the questionnaire.
2. Find the passage supporting each supplied answer.
3. Copy the original questionnaire text as accurately
   as possible.
4. Do not invent or paraphrase evidence.
5. Keep evidence concise.
6. If an answer is null, return null evidence.
7. If supporting text cannot be found, return null.
8. Do not change any extracted answers.
9. Return valid JSON matching the supplied schema.
"""

    user_prompt = f"""
EXTRACTED ANSWERS:

{json.dumps(
    answer_data,
    indent=2,
    ensure_ascii=False
)}


ORIGINAL QUESTIONNAIRE:

{questionnaire_text}


TASK:

For every non-null answer, find the original
questionnaire passage that supports it.

Return the shortest complete supporting passage.

If an answer is null, return null evidence.

Do not generate or modify any answers.

Return only the evidence JSON.
"""

    print("\n========== QUESTIONNAIRE EVIDENCE EXTRACTION ==========")

    try:

        raw_output = call_ollama(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            response_format=schema,
            num_predict=4096
        )

        print("\n========== RAW EVIDENCE ==========")

        print(raw_output)

        evidence_data = json.loads(raw_output)

        return normalize_null_values(evidence_data)

    except Exception as error:

        print("\nEvidence extraction failed:", error)

        print("Original answers will be preserved.")

        return {
            section: {
                field: {
                    "evidence": None
                }
                for field in fields
            }
            for section, fields in get_field_definitions().items()
        }


# ============================================================
# EVIDENCE VERIFICATION
# ============================================================

def normalize_text(text):

    if not isinstance(text, str):
        return ""

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip().lower()


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

    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split()
    )


def answer_is_grounded(answer, source_text, field_name):
    """
    The answer may only contain facts that exist in the questionnaire.

    - every number must appear in the questionnaire
    - 60% of the answer words (40% for summary fields) must appear

    This removes values copied from prompt/schema examples or taken
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


def verify_questionnaire_evidence(
    extracted_data,
    evidence_data,
    questionnaire_text
):

    verified_evidence = {
        "company": {},
        "product": {}
    }

    normalized_questionnaire = normalize_text(
        questionnaire_text
    )

    answer_data = extracted_data.model_dump()

    for section, fields in answer_data.items():

        for field_name, answer in fields.items():

            quotation = (
                evidence_data
                .get(section, {})
                .get(field_name, {})
                .get("evidence")
            )

            if (
                (section == "company" and field_name == "company_name")
                or
                (section == "product" and field_name == "product_name")
            ):

                status = "user_input"
                source = "user_input"
                quotation = None

            elif answer is None:

                status = "missing"
                source = None
                quotation = None

            elif not answer_is_grounded(
                answer,
                questionnaire_text,
                field_name
            ):

                # The answer contains values that do not exist in
                # the questionnaire (copied example / invented).
                # Remove it so the web search can fill the field.
                print(
                    f"Rejected ungrounded answer: "
                    f"{section}.{field_name} -> {answer}"
                )

                setattr(
                    getattr(extracted_data, section),
                    field_name,
                    None
                )

                status = "rejected_not_in_questionnaire"
                source = None
                quotation = None

            elif not isinstance(quotation, str) or not quotation.strip():

                status = "evidence_not_found"
                source = "questionnaire"

            elif normalize_text(quotation) in normalized_questionnaire:

                status = "verified"
                source = "questionnaire"

            else:

                status = "needs_review"
                source = "questionnaire"

            verified_evidence[section][field_name] = {
                "evidence": quotation,
                "source": source,
                "verification_status": status
            }

    return extracted_data, verified_evidence


# ============================================================
# MAIN FUNCTION
# ============================================================

def extract_from_questionnaire(
    questionnaire_text: str,
    company_name: str,
    product_name: str,
    return_evidence: bool = False
):

    if not questionnaire_text or not questionnaire_text.strip():

        raise ValueError(
            "Questionnaire text is empty."
        )

    # STEP 1: Original answer extraction
    extracted_data = extract_questionnaire_answers(
        questionnaire_text=questionnaire_text,
        company_name=company_name,
        product_name=product_name
    )

    # STEP 2: Separate evidence extraction
    evidence_data = extract_questionnaire_evidence(
        extracted_data=extracted_data,
        questionnaire_text=questionnaire_text
    )

    # STEP 3: Verify evidence only
    extracted_data, verified_evidence = (
        verify_questionnaire_evidence(
            extracted_data=extracted_data,
            evidence_data=evidence_data,
            questionnaire_text=questionnaire_text
        )
    )

    print("\n========== FINAL QUESTIONNAIRE JSON ==========")

    print(
        extracted_data.model_dump_json(
            indent=4
        )
    )

    print("\n========== VERIFIED QUESTIONNAIRE EVIDENCE ==========")

    print(
        json.dumps(
            verified_evidence,
            indent=4,
            ensure_ascii=False
        )
    )

    if return_evidence:

        return extracted_data, verified_evidence

    return extracted_data
