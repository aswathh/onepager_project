 
import asyncio
from urllib.parse import urlparse

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode
)


# Pages with less text than this are treated as failed (login walls,
# empty JS shells) and replaced by the search snippet.
MIN_USEFUL_CHARACTERS = 300

# Crawl the highest-priority pages first and stop at this many.
MAX_PAGES = 30


# ============================================================
# STEP 1: Extract markdown from Crawl4AI response
# ============================================================

def extract_markdown(result):
    """
    Handle different Crawl4AI markdown response formats.

    raw_markdown is preferred: fit_markdown prunes "low value" blocks
    and often removes short facts (employee count, headquarters,
    customer/partner names).
    """

    markdown = getattr(result, "markdown", None)

    if markdown is None:
        return ""

    raw = getattr(markdown, "raw_markdown", None)

    if isinstance(raw, str) and raw.strip():
        return raw

    fit = getattr(markdown, "fit_markdown", None)

    if isinstance(fit, str) and fit.strip():
        return fit

    if isinstance(markdown, str):
        return markdown

    return ""


# ============================================================
# STEP 2: Extract domain from URL
# ============================================================

def get_domain(url):
    """
    Extract domain from a URL.
    """

    if not url:
        return ""

    return urlparse(url).netloc.lower().removeprefix("www.")


# ============================================================
# STEP 3: Crawl one webpage
# ============================================================

def snippet_fallback(page, error):
    """
    Use the search-result title and snippet when the page itself
    cannot be read (LinkedIn login wall, blocked site, timeout).
    Snippets often contain exactly the fact that was searched.
    """

    snippet = (page.get("snippet") or "").strip()
    title = (page.get("title") or "").strip()

    content = f"{title}\n{snippet}".strip() if snippet else ""

    return {
        **page,
        "final_url": page.get("url", ""),
        "success": bool(content),
        "content": content,
        "content_length": len(content),
        "content_source": "search_snippet" if content else None,
        "error": error
    }


async def crawl_single_url(crawler, page, run_config):
    """
    Crawl one webpage while preserving its search and
    company-identity metadata.
    """

    url = page.get("url", "")

    try:
        result = await crawler.arun(
            url=url,
            config=run_config
        )

        if not result.success:
            return snippet_fallback(
                page,
                getattr(result, "error_message", "Crawling failed")
            )

        content = extract_markdown(result)

        if len(content.strip()) < MIN_USEFUL_CHARACTERS:

            fallback = snippet_fallback(
                page,
                "Page text too short (login wall or empty page)"
            )

            # Keep whichever text is longer.
            if len(fallback["content"]) > len(content.strip()):
                return fallback

        final_url = getattr(result, "url", url)

        return {
            **page,
            "final_url": final_url,
            "success": True,
            "content": content,
            "content_length": len(content),
            "content_source": "page",
            "error": None
        }

    except Exception as error:

        return snippet_fallback(page, str(error))


# ============================================================
# STEP 4: Collect unique verified URLs
# ============================================================

def collect_unique_pages(search_results):
    """
    Collect unique URLs from identity-filtered search results.

    Preserve:
    - Company identity verification
    - Source type and priority
    - Related missing fields
    - Search result metadata
    """

    unique_pages = {}

    for field, field_data in search_results.items():

        for item in field_data.get("results", []):

            url = item.get("url")

            if not url:
                continue

            # Only crawl results accepted by company identity filter.
            if item.get("identity_verified") is not True:
                print(
                    "\nSkipping unverified webpage:",
                    url
                )
                continue

            if url not in unique_pages:

                unique_pages[url] = {
                    "title": item.get("title", ""),
                    "url": url,
                    "domain": (
                        item.get("domain")
                        or get_domain(url)
                    ),
                    "snippet": item.get("snippet", ""),

                    # Identity metadata
                    "identity_verified": True,
                    "identity_reason": item.get(
                        "identity_reason", ""
                    ),

                    # Source metadata
                    "source_type": item.get(
                        "source_type", "other"
                    ),
                    "source_priority": item.get(
                        "source_priority", 4
                    ),

                    # Missing fields associated with this URL
                    "related_fields": []
                }

            # Associate one webpage with multiple missing fields.
            if field not in unique_pages[url]["related_fields"]:
                unique_pages[url]["related_fields"].append(field)

    return list(unique_pages.values())


# ============================================================
# STEP 5: Crawl all unique webpages
# ============================================================

async def crawl_webpages(search_results):
    """
    Crawl unique, identity-verified URLs.

    Returns crawled webpage data with identity and source
    metadata preserved for downstream processing.
    """

    unique_pages = collect_unique_pages(search_results)

    # Official website first, then LinkedIn, then others.
    unique_pages.sort(
        key=lambda item: item.get("source_priority", 4)
    )

    unique_pages = unique_pages[:MAX_PAGES]

    print("\nUnique verified URLs to crawl:", len(unique_pages))

    if not unique_pages:
        print(
            "\nNo identity-verified webpages available for crawling."
        )
        return []

    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )

    # word_count_threshold=30 used to delete every text block shorter
    # than 30 words, which removed short facts such as
    # "Employees: 25" or "Headquarters: <city>" and logo/name lists.
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=0,
        page_timeout=60000,
        delay_before_return_html=1.5,
        remove_overlay_elements=True
    )

    crawled_pages = []

    async with AsyncWebCrawler(
        config=browser_config
    ) as crawler:

        for index, page in enumerate(
            unique_pages,
            start=1
        ):

            print(
                f"\n[{index}/{len(unique_pages)}] Crawling:"
            )
            print(page["url"])

            crawl_result = await crawl_single_url(
                crawler,
                page,
                run_config
            )

            print("Success:", crawl_result["success"])
            print(
                "Content characters:",
                crawl_result["content_length"]
            )
            print(
                "Identity verified:",
                crawl_result["identity_verified"]
            )
            print(
                "Related fields:",
                crawl_result["related_fields"]
            )

            crawled_pages.append(crawl_result)

    return crawled_pages


# ============================================================
# STEP 6: Synchronous entry point
# ============================================================

def crawl_search_results(search_results):
    """
    Synchronous entry point for the crawler.
    """

    return asyncio.run(
        crawl_webpages(search_results)
    )
