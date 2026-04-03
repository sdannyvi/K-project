"""
Download J. Krishnamurti transcript pages into plain-text files under K-texts/.

Default: discover transcript URLs via the Internet Archive CDX API and fetch each
page from Wayback (HTTP), avoiding Cloudflare on the live site. Optional --live
uses Playwright against jkrishnamurti.org (often blocked). Parses Drupal-style
fields for title, subtitle/metadata, and body.

Update when jkrishnamurti.org HTML structure, CDX usage, or search URL parameters change.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

import k_config

logger = logging.getLogger(__name__)

# Wayback CDX often returns 502/503/504 on heavy queries; retry before failing.
_CDX_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 502, 503, 504})
_CDX_MAX_RETRIES: int = 8
_CDX_RETRY_BACKOFF_BASE_S: float = 5.0
_CDX_HTTP_TIMEOUT_S: float = 300.0
# Stop CDX pagination after this many consecutive batches with no new canonical URLs.
_CDX_ZERO_STREAK_STOP: int = 1

# Lines or phrases to strip from extracted text (share UI, icons).
_NOISE_SUBSTRINGS: tuple[str, ...] = (
    "Facebook icon",
    "Twitter icon",
    "LinkedIn icon",
    "Email icon",
)


def _strip_noise(text: str) -> str:
    """
    Remove known social-share noise and collapse excessive blank lines.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text string.
    """
    for s in _NOISE_SUBSTRINGS:
        text = text.replace(s, "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_content_url(base: str, href: str | None) -> str | None:
    """
    Turn a search-result href into an absolute canonical transcript URL.

    Args:
        base: Site origin, e.g. https://jkrishnamurti.org
        href: Raw href from an anchor (may be relative or wayback-prefixed).

    Returns:
        Absolute URL ending with / for /content/... paths, else None.
    """
    if not href or href.startswith("#"):
        return None
    if "web.archive.org" in href:
        m = re.search(r"https?://[^/]+/content/[^?\s\"']+", href)
        if m:
            href = m.group(0)
        else:
            return None
    full = urljoin(base + "/", href)
    parsed = urlparse(full)
    if "/content/" not in parsed.path:
        return None
    path = parsed.path.rstrip("/") + "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def canonical_content_url(original: str) -> str | None:
    """
    Normalize a CDX or browser URL to a canonical https transcript URL.

    Args:
        original: Raw URL from CDX (may include query strings or encoding).

    Returns:
        https://jkrishnamurti.org/content/<slug>/ or None if not a single transcript.
    """
    u = unquote(original.strip())
    m = re.match(
        r"https?://(?:www\.)?jkrishnamurti\.org/content/([a-z0-9-]+)/?(?:\?.*)?$",
        u,
        re.I,
    )
    if not m:
        return None
    slug = m.group(1)
    return f"https://jkrishnamurti.org/content/{slug}/"


def _cdx_get(client: httpx.Client, api: str, params: dict[str, str]) -> httpx.Response:
    """
    GET the CDX API with retries on transient errors (504 Gateway Timeout, etc.).

    Args:
        client: HTTP client.
        api: CDX endpoint URL.
        params: Query parameters.

    Returns:
        Successful HTTP response.

    Raises:
        httpx.HTTPStatusError: If all retries fail or status is non-retryable.
    """
    last_status: int | None = None
    for attempt in range(_CDX_MAX_RETRIES):
        resp = client.get(api, params=params)
        last_status = resp.status_code
        if resp.status_code not in _CDX_RETRYABLE_STATUSES:
            resp.raise_for_status()
            return resp
        if attempt == _CDX_MAX_RETRIES - 1:
            logger.error(
                "CDX failed after %s attempts (last status %s)",
                _CDX_MAX_RETRIES,
                last_status,
            )
            resp.raise_for_status()
        wait_s = min(_CDX_RETRY_BACKOFF_BASE_S * (2**attempt), 120.0)
        logger.warning(
            "CDX returned %s (attempt %s/%s); retrying in %.0fs",
            resp.status_code,
            attempt + 1,
            _CDX_MAX_RETRIES,
            wait_s,
        )
        time.sleep(wait_s)
    raise RuntimeError("_cdx_get: exhausted retries without return")


def discover_urls_cdx(max_urls: int | None) -> list[str]:
    """
    Paginate the Wayback CDX API for unique /content/<slug>/ transcript URLs.

    Stops early after one CDX batch that adds no new canonical URLs (then proceeds
    to download).

    Args:
        max_urls: Stop after this many URLs, or None for full crawl.

    Returns:
        Sorted list of canonical transcript URLs.
    """
    seen: set[str] = set()
    offset = 0
    zero_streak = 0
    page_size = 5000
    api = "https://web.archive.org/cdx/search/cdx"
    with httpx.Client(timeout=_CDX_HTTP_TIMEOUT_S, follow_redirects=True) as client:
        while max_urls is None or len(seen) < max_urls:
            params = {
                "url": k_config.WAYBACK_CDX_CONTENT_PREFIX,
                "matchType": "prefix",
                "output": "json",
                "fl": "original",
                "collapse": "urlkey",
                "limit": str(page_size),
                "offset": str(offset),
            }
            resp = _cdx_get(client, api, params)
            data = resp.json()
            if not data or len(data) < 2:
                break
            rows = data[1:]
            added = 0
            for row in rows:
                u = row[0] if isinstance(row, list) else row
                c = canonical_content_url(u)
                if c and c not in seen:
                    seen.add(c)
                    added += 1
                    if max_urls is not None and len(seen) >= max_urls:
                        break
            logger.info(
                "CDX offset %s: +%s new (total unique %s)",
                offset,
                added,
                len(seen),
            )
            if added > 0:
                zero_streak = 0
            else:
                zero_streak += 1
            if zero_streak >= _CDX_ZERO_STREAK_STOP:
                logger.info(
                    "CDX: batch with +0 new (streak >= %s); stopping discovery (total unique %s).",
                    _CDX_ZERO_STREAK_STOP,
                    len(seen),
                )
                break
            if max_urls is not None and len(seen) >= max_urls:
                break
            if len(rows) < page_size:
                break
            offset += page_size
    return sorted(seen)


def discover_urls_playwright(
    max_pages: int,
    base: str,
    search_path: str,
    search_query: str,
    delay_s: float,
) -> list[str]:
    """
    Paginate JK search results and collect unique transcript /content/ URLs.

    Args:
        max_pages: Maximum search page index to request.
        base: Site origin URL.
        search_path: Path for search, e.g. /jksearch.
        search_query: Query string without leading '?'.
        delay_s: Pause between page loads.

    Returns:
        Sorted list of unique absolute transcript URLs.

    Raises:
        RuntimeError: If Playwright is not installed or browser launch fails.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "Playwright is required for scraping the live site. "
            "Install with: pip install playwright && playwright install chromium"
        ) from e

    seen: set[str] = set()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()
        for pg in range(1, max_pages + 1):
            url = f"{base.rstrip('/')}{search_path}?{search_query}&page={pg}"
            logger.info("Loading search page %s: %s", pg, url)
            page.goto(url, wait_until="domcontentloaded", timeout=120_000)
            time.sleep(max(delay_s, 2.0))
            html = page.content()
            if "Just a moment" in html and "challenge-platform" in html:
                logger.warning(
                    "Cloudflare challenge page — use default Wayback mode or --urls-file."
                )
            before = len(seen)
            for u in extract_content_links_from_html(base, html):
                seen.add(u)
            added = len(seen) - before
            logger.info("Page %s: +%s new URLs (total unique %s)", pg, added, len(seen))
            if added == 0:
                logger.info("No new URLs on page %s; stopping pagination.", pg)
                break
        browser.close()
    return sorted(seen)


def extract_content_links_from_html(base: str, html: str) -> list[str]:
    """
    Parse search result HTML and return canonical /content/ transcript URLs.

    Args:
        base: Site origin for resolving relative links.
        html: Full HTML document.

    Returns:
        List of unique absolute URLs.
    """
    soup = BeautifulSoup(html, "lxml")
    out: set[str] = set()
    for a in soup.select('a[href*="/content/"]'):
        nu = normalize_content_url(base, a.get("href"))
        if nu:
            out.add(nu)
    return list(out)


def parse_transcript_html(html: str) -> tuple[str, str, str]:
    """
    Extract title, metadata line, and body text from a transcript page.

    Args:
        html: Full HTML of a single transcript page.

    Returns:
        (title, metadata_line, body_text). Missing parts become empty strings.
    """
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.select_one(
        ".field--name-title h1, .field--name-title h2, .field--name-title h3"
    )
    title = _strip_noise(title_el.get_text(" ", strip=True)) if title_el else ""
    if not title:
        og = soup.select_one('meta[property="og:title"]')
        if og and og.get("content"):
            title = _strip_noise(og["content"].strip())

    meta_el = soup.select_one(".field--name-field-title-bottom")
    meta = _strip_noise(meta_el.get_text(" ", strip=True)) if meta_el else ""

    body_el = soup.select_one(".field--name-body")
    if body_el:
        body = _strip_noise(body_el.get_text("\n", strip=False))
    else:
        body = ""

    return title, meta, body


def format_k_text_file(title: str, metadata: str, body: str) -> str:
    """
    Build the canonical plain-text layout for K-texts/*.txt.

    Args:
        title: Talk title (line 1).
        metadata: Subtitle / location / date line.
        body: Full transcript.

    Returns:
        File contents as a single string.
    """
    parts = [title, "", metadata, "", body]
    return "\n".join(parts).rstrip() + "\n"


def slug_from_content_url(url: str) -> str:
    """
    Derive a filesystem-safe filename stem from a /content/... URL.

    Args:
        url: Canonical transcript URL.

    Returns:
        Slug string safe for use as a filename stem.
    """
    path = urlparse(url).path.strip("/").split("/")
    if len(path) >= 2 and path[0] == "content":
        raw = path[1]
    else:
        raw = re.sub(r"[^\w-]", "_", url)[:80]
    safe = re.sub(r"[^\w\-]+", "-", raw).strip("-")[:180]
    return safe or "document"


def write_discovered_urls_file(path: Path, urls: list[str]) -> None:
    """
    Write one canonical transcript URL per line for reuse without re-running CDX.

    Args:
        path: Output file path (e.g. discovered_urls.txt).
        urls: Sorted or unsorted list of URLs.
    """
    path.write_text("\n".join(urls) + ("\n" if urls else ""), encoding="utf-8")


def load_completed_urls(path: Path) -> set[str]:
    """
    Load URLs already saved successfully (resume manifest).

    Args:
        path: Path to .completed_urls.txt.

    Returns:
        Set of canonical URL strings.
    """
    if not path.is_file():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def append_completed_url(path: Path, url: str) -> None:
    """
    Append one URL to the completed manifest after a successful write.

    Args:
        path: Append-only file path (parent dirs created if needed).
        url: Canonical transcript URL that was saved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(url + "\n")


def load_urls_file(path: Path) -> list[str]:
    """
    Load one transcript URL per non-empty non-comment line.

    Args:
        path: Path to a UTF-8 text file.

    Returns:
        List of URL strings with whitespace stripped.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def fetch_html_wayback(canonical_url: str, snapshot: str) -> str | None:
    """
    Fetch archived transcript HTML from the Wayback Machine (plain HTTP).

    Args:
        canonical_url: e.g. https://jkrishnamurti.org/content/foo/
        snapshot: Wayback timestamp string (14 digits).

    Returns:
        HTML document text, or None if no variant returns 200 with transcript fields.
    """
    variants = [
        canonical_url.replace(
            "https://jkrishnamurti.org/", "https://www.jkrishnamurti.org/", 1
        ),
        canonical_url,
    ]
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for orig in variants:
            wb = f"https://web.archive.org/web/{snapshot}id_/{orig}"
            try:
                r = client.get(wb)
            except httpx.RequestError as exc:
                logger.warning("Wayback HTTP error for %s: %s", wb[:80], exc)
                continue
            if r.status_code != 200:
                continue
            if "field--name-body" in r.text or "field--name-title" in r.text:
                return r.text
    return None


def fetch_html_playwright(url: str) -> str:
    """
    Fetch a single URL and return HTML after Cloudflare/JS load.

    Args:
        url: Full page URL.

    Returns:
        Document HTML string.

    Raises:
        RuntimeError: If Playwright is unavailable.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "Playwright is required. pip install playwright && playwright install chromium"
        ) from e

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        time.sleep(1.0)
        html = page.content()
        browser.close()
    return html


def save_transcripts(
    urls: Iterable[str],
    out_dir: Path,
    delay_s: float,
    fetch_mode: str = "wayback",
    resume: bool = True,
    completed_urls_path: Path | None = None,
) -> int:
    """
    Fetch each transcript URL and write K-texts/<slug>.txt.

    Args:
        urls: Iterable of canonical transcript URLs.
        out_dir: Directory K-texts (created if missing).
        delay_s: Delay between fetches.
        fetch_mode: "wayback" (HTTP archive) or "live" (Playwright).
        resume: If True, skip URLs listed in completed manifest or existing non-empty .txt.
        completed_urls_path: Append-only file of successful URLs (default: k_config).

    Returns:
        Number of files written this run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = completed_urls_path or k_config.SCRAPE_COMPLETED_URLS_FILE
    completed: set[str] = load_completed_urls(done_path) if resume else set()
    n = 0
    for url in urls:
        slug = slug_from_content_url(url)
        dest = out_dir / f"{slug}.txt"
        if resume:
            if url in completed:
                logger.info("Resume: skip (completed list) %s", dest.name)
                continue
            if dest.is_file() and dest.stat().st_size > 0:
                logger.info("Resume: skip (file exists) %s", dest.name)
                if url not in completed:
                    append_completed_url(done_path, url)
                    completed.add(url)
                continue
        logger.info("Fetching %s -> %s", url, dest.name)
        if fetch_mode == "wayback":
            html = fetch_html_wayback(url, k_config.WAYBACK_SNAPSHOT)
            if html is None:
                logger.warning(
                    "Wayback: no usable archived HTML for %s — skipping.",
                    url,
                )
                time.sleep(delay_s)
                continue
        else:
            try:
                html = fetch_html_playwright(url)
            except Exception as exc:
                logger.warning(
                    "Live fetch failed for %s (%s) — skipping.",
                    url,
                    exc,
                )
                time.sleep(delay_s)
                continue
        title, meta, body = parse_transcript_html(html)
        if not body and not title:
            logger.warning("No title/body parsed for %s — skipping write.", url)
            time.sleep(delay_s)
            continue
        text = format_k_text_file(title, meta, body)
        dest.write_text(text, encoding="utf-8")
        append_completed_url(done_path, url)
        completed.add(url)
        n += 1
        time.sleep(delay_s)
    return n


def main() -> None:
    """CLI: discover URLs and/or download transcripts into K-texts/."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download JK transcripts into K-texts/")
    parser.add_argument(
        "--urls-file",
        type=Path,
        help="Optional file with one transcript URL per line (skip search discovery).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=k_config.SCRAPE_MAX_PAGES,
        help="Max search pages to crawl when discovering URLs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=k_config.K_TEXTS_DIR,
        help="Output directory for .txt files (default: K-texts/).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only print discovered URLs, do not fetch transcripts.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Discover/fetch from live site with Playwright (often blocked by Cloudflare).",
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=None,
        help="Cap discovered URLs (Wayback CDX mode only). Default: no cap.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-fetch every URL; ignore .completed_urls.txt and overwrite existing files.",
    )
    parser.add_argument(
        "--discovered-out",
        type=Path,
        default=None,
        help=f"Where to write discovered URL list (default: {k_config.DISCOVERED_URLS_FILE}).",
    )
    args = parser.parse_args()

    base = k_config.JK_SITE_BASE
    if args.urls_file:
        urls = load_urls_file(args.urls_file)
        logger.info("Loaded %s URLs from %s", len(urls), args.urls_file)
    elif args.live:
        urls = discover_urls_playwright(
            max_pages=args.max_pages,
            base=base,
            search_path=k_config.JK_SEARCH_PATH,
            search_query=k_config.JK_SEARCH_QUERY,
            delay_s=k_config.SCRAPE_DELAY_SECONDS,
        )
        logger.info("Discovered %s unique transcript URLs (live)", len(urls))
    else:
        urls = discover_urls_cdx(args.max_urls)
        logger.info("Discovered %s unique transcript URLs (Wayback CDX)", len(urls))

    discovered_path = args.discovered_out or k_config.DISCOVERED_URLS_FILE
    if urls and not args.urls_file:
        write_discovered_urls_file(discovered_path, urls)
        logger.info("Wrote %s URL(s) to %s", len(urls), discovered_path)

    if args.skip_download:
        for u in urls:
            print(u)
        return

    fetch_mode = "live" if args.live else "wayback"
    written = save_transcripts(
        urls,
        args.out_dir,
        delay_s=k_config.SCRAPE_DELAY_SECONDS,
        fetch_mode=fetch_mode,
        resume=not args.no_resume,
    )
    logger.info("Wrote %s transcript files under %s", written, args.out_dir)


if __name__ == "__main__":
    main()
