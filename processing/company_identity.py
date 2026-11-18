
import re
from difflib import SequenceMatcher


# ============================================================
# Text utilities
# ============================================================

def normalize_text(value):
    """Normalize text for company-name comparison."""

    if not value:
        return ""

    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)

    return " ".join(value.split())


LEGAL_SUFFIXES = {
    "inc", "incorporated", "ltd", "limited", "llc", "llp", "plc",
    "corp", "corporation", "gmbh", "pvt", "private", "co", "company"
}


def get_name_variants(company_name):
    """
    Return normalized spellings of the company name.

    "Acme Pvt Ltd" -> ["acme pvt ltd", "acme"]
    A page that says "Acme" must still match.
    """

    full = normalize_text(company_name)

    if not full:
        return []

    variants = [full]

    tokens = full.split()

    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens = tokens[:-1]

    short = " ".join(tokens)

    if short and short not in variants:
        variants.append(short)

    return variants


def contains_exact_name(text, company_name):
    """
    Check whether the exact company name appears in text.

    Example:
        Assessli matches Assessli
        Assessli does not match Assessio
    """

    normalized_text = normalize_text(text)

    if not normalized_text:
        return False

    for variant in get_name_variants(company_name):

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(variant)
            + r"(?![a-z0-9])"
        )

        if re.search(pattern, normalized_text):
            return True

    return False


def name_in_url(url, domain, company_name):
    """
    Check whether the company name is part of the domain or URL path.

    assessli.com                         -> True
    linkedin.com/company/assessli/       -> True

    Search snippets are often short and do not repeat the company
    name, so the URL is a second, reliable signal.
    """

    compact_url = re.sub(r"[^a-z0-9]", "", (url or "").lower())
    compact_domain = re.sub(r"[^a-z0-9]", "", (domain or "").lower())

    for variant in get_name_variants(company_name):

        compact_name = variant.replace(" ", "")

        if len(compact_name) < 4:
            continue

        if compact_name in compact_domain:
            return True

        # Only the path of a social/profile URL, e.g. /company/<name>
        if "/company/" in (url or "").lower() and compact_name in compact_url:
            return True

    return False


# ============================================================
# Similar company-name detection
# ============================================================

def find_similar_company_names(text, company_name):
    """
    Find words that closely resemble the requested company name.

    This helps detect cases such as:
        Assessli vs Assessio

    It is a warning mechanism, not proof of identity.
    """

    normalized_company = normalize_text(company_name)

    if len(normalized_company.replace(" ", "")) < 5:
        return []

    words = set(normalize_text(text).split())

    similar_names = []

    for word in words:

        if len(word) < 5:
            continue

        similarity = SequenceMatcher(
            None,
            normalized_company.replace(" ", ""),
            word
        ).ratio()

        if similarity >= 0.80 and word != normalized_company:
            similar_names.append({
                "name": word,
                "similarity": round(similarity, 2)
            })

    return similar_names


SOCIAL_OR_LISTING_DOMAINS = (
    "linkedin.com", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "youtube.com", "wikipedia.org", "crunchbase.com",
    "glassdoor.com", "g2.com", "capterra.com", "pitchbook.com"
)


def is_social_or_listing(domain):
    """Domains whose name never proves ownership of the company."""

    domain = (domain or "").lower()

    return any(
        domain == item or domain.endswith("." + item)
        for item in SOCIAL_OR_LISTING_DOMAINS
    )


# ============================================================
# Company identity verification
# ============================================================

def verify_company_identity(
    company_name,
    result,
    official_domain=None
):
    """
    Verify whether a search result relates to the requested company.

    Returns:
        {
            "verified": True/False,
            "reason": "...",
            "source_type": "..."
        }

    Verification rules:

    1. Verified official domain:
       Accept as official-domain evidence.

    2. LinkedIn:
       Require the exact company name in title or snippet.

    3. Other websites:
       Require the exact company name in title or snippet.

    4. Similar company names:
       Reject when only a similar name is found.
    """

    title = result.get("title", "")
    snippet = result.get("snippet", "")
    url = result.get("url", "")
    domain = result.get("domain", "")

    text_to_check = f"{title} {snippet}"

    # --------------------------------------------------------
    # Normalize domain
    # --------------------------------------------------------

    domain = (domain or "").lower().strip()

    if not domain and url:

        from search.ddgs_search import get_domain

        domain = get_domain(url)

    if domain.startswith("www."):
        domain = domain[4:]

    # --------------------------------------------------------
    # Check official domain
    # --------------------------------------------------------

    if official_domain:

        official_domain = official_domain.lower().strip()

        official_domain = official_domain.replace(
            "https://", ""
        ).replace("http://", "")

        official_domain = official_domain.split("/")[0]

        if official_domain.startswith("www."):
            official_domain = official_domain[4:]

        if (
            domain == official_domain
            or domain.endswith("." + official_domain)
        ):

            return {
                "verified": True,
                "reason": "Result belongs to the verified official domain.",
                "source_type": "official_website"
            }

    # --------------------------------------------------------
    # Exact company-name check
    # --------------------------------------------------------

    exact_match = contains_exact_name(
        text_to_check,
        company_name
    )

    # The company name inside the domain/URL is also valid evidence.
    url_match = name_in_url(url, domain, company_name)

    if url_match and not is_social_or_listing(domain):
        exact_match = True

    # --------------------------------------------------------
    # LinkedIn verification
    # --------------------------------------------------------

    is_linkedin = (
        domain == "linkedin.com"
        or domain.endswith(".linkedin.com")
    )

    if is_linkedin:

        if exact_match or url_match:

            return {
                "verified": True,
                "reason": "Exact company name found on LinkedIn.",
                "source_type": "linkedin"
            }

        return {
            "verified": False,
            "reason": "LinkedIn result does not identify the requested company.",
            "source_type": "linkedin"
        }

    # --------------------------------------------------------
    # Other websites
    # --------------------------------------------------------

    if exact_match:

        return {
            "verified": True,
            "reason": "Exact company name found in title or snippet.",
            "source_type": "other"
        }

    # --------------------------------------------------------
    # Detect similar company names
    # --------------------------------------------------------

    similar_names = find_similar_company_names(
        text_to_check,
        company_name
    )

    if similar_names:

        return {
            "verified": False,
            "reason": (
                "Possible similarly named company detected: "
                + ", ".join(
                    item["name"] for item in similar_names
                )
            ),
            "source_type": "other"
        }

    return {
        "verified": False,
        "reason": "Requested company name was not found in the result.",
        "source_type": "other"
    }


# ============================================================
# Filter complete search output
# ============================================================

def filter_search_results_by_identity(
    search_output,
    company_name,
    official_domain=None
):
    """
    Filter search results grouped by missing field.

    Preserves the existing search-output structure.

    Accepted results remain under:
        results

    Rejected results are stored under:
        rejected_results
    """

    filtered_output = {}

    print("\n========== COMPANY IDENTITY VERIFICATION ==========")

    for field, field_data in search_output.items():

        accepted_results = []
        rejected_results = []

        results = field_data.get("results", [])

        print(f"\nChecking field: {field}")

        for result in results:

            verification = verify_company_identity(
                company_name=company_name,
                result=result,
                official_domain=official_domain
            )

            updated_result = result.copy()

            updated_result["identity_verified"] = (
                verification["verified"]
            )

            updated_result["identity_reason"] = (
                verification["reason"]
            )

            updated_result["source_type"] = (
                verification["source_type"]
            )

            if verification["verified"]:

                accepted_results.append(updated_result)

                print(
                    "[ACCEPTED]",
                    result.get("title", "")
                )

            else:

                rejected_results.append(updated_result)

                print(
                    "[REJECTED]",
                    result.get("title", "")
                )

                print(
                    "Reason:",
                    verification["reason"]
                )

        filtered_output[field] = {
            **field_data,
            "results": accepted_results,
            "rejected_results": rejected_results
        }

        print("Accepted:", len(accepted_results))
        print("Rejected:", len(rejected_results))

    return filtered_output
