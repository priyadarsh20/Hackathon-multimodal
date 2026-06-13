"""
playwright_scraper.py
Scrapes reference design images from Dribbble / Mobbin using Playwright.
These images are embedded and stored in the Image RAG index for similarity search.

Usage:
  scraper = DesignScraper()
  results = await scraper.scrape_references(sources=["dribbble"])
  # returns list of {url, title, source, description}

Falls back gracefully if Playwright browser is not installed.
"""

from __future__ import annotations
import asyncio
import base64
from typing import Optional
import requests


# ── Reference image database (used when Playwright unavailable) ───────────────
# Curated public design reference URLs as a static fallback
REFERENCE_DESIGNS = [
    {
        "title": "SaaS Dashboard — dark sidebar",
        "description": "Left sidebar navigation, top search bar, KPI cards in a grid, data table below, dark/light mode toggle. Standard SaaS admin layout.",
        "source": "design_reference",
        "category": "dashboard",
        "pattern_tags": ["sidebar", "dark-mode", "data-table", "kpi-cards"],
    },
    {
        "title": "Mobile checkout flow",
        "description": "Bottom sheet cart, product thumbnail, price anchoring with strikethrough, sticky CTA button at bottom, progress indicator top.",
        "source": "design_reference",
        "category": "ecommerce",
        "pattern_tags": ["checkout", "mobile", "bottom-sheet", "sticky-cta"],
    },
    {
        "title": "Onboarding — value-first",
        "description": "3-step onboarding, large illustration, single headline per screen, skip option top-right, progress dots, primary CTA centered.",
        "source": "design_reference",
        "category": "onboarding",
        "pattern_tags": ["onboarding", "illustration", "progress-dots", "value-proposition"],
    },
    {
        "title": "Landing page — SaaS B2B",
        "description": "Hero with product screenshot, social proof logos row, feature grid 3-column, testimonial carousel, pricing table, FAQ accordion.",
        "source": "design_reference",
        "category": "landing",
        "pattern_tags": ["hero", "social-proof", "pricing", "feature-grid"],
    },
    {
        "title": "Mobile feed — social app",
        "description": "Bottom tab navigation 5 items, card feed with avatar, image, text snippet, like/comment/share row. Infinite scroll pattern.",
        "source": "design_reference",
        "category": "social",
        "pattern_tags": ["bottom-nav", "feed", "social-actions", "avatar"],
    },
    {
        "title": "Settings page — profile",
        "description": "Grouped list sections with headers, toggle switches for notifications, chevron rows for deep navigation, destructive actions in red at bottom.",
        "source": "design_reference",
        "category": "settings",
        "pattern_tags": ["settings", "toggle", "grouped-list", "destructive-action"],
    },
    {
        "title": "Dashboard analytics — light",
        "description": "Top metric cards with sparklines, area chart with date picker, segmented control, filterable data table, export button.",
        "source": "design_reference",
        "category": "analytics",
        "pattern_tags": ["metrics", "chart", "sparkline", "data-table", "filters"],
    },
    {
        "title": "Auth screen — sign up",
        "description": "Centered card, logo top, OAuth buttons (Google/GitHub), divider OR, email/password fields, terms checkbox, submit CTA, sign-in link below.",
        "source": "design_reference",
        "category": "auth",
        "pattern_tags": ["auth", "oauth", "centered-card", "form"],
    },
    {
        "title": "Product detail — mobile",
        "description": "Image carousel with dots, sticky header on scroll, color/size variant selectors, accordion for description, related products row, sticky bottom bar with Add to Cart.",
        "source": "design_reference",
        "category": "ecommerce",
        "pattern_tags": ["product-detail", "carousel", "variants", "sticky-bar"],
    },
    {
        "title": "Empty state — first run",
        "description": "Centered illustration (200px), headline, supporting body text, primary CTA button, optional secondary link. Used for empty feeds, no results, new user.",
        "source": "design_reference",
        "category": "empty-state",
        "pattern_tags": ["empty-state", "illustration", "first-run", "cta"],
    },
]


class DesignScraper:
    """
    Playwright-based scraper for reference design images.
    Gracefully degrades to static reference database if browser unavailable.
    """

    def __init__(self):
        self._playwright_available = self._check_playwright()

    def _check_playwright(self) -> bool:
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                browser.close()
            return True
        except Exception:
            return False

    async def scrape_references(
        self,
        sources: list[str] | None = None,
        max_per_source: int = 20,
    ) -> list[dict]:
        """
        Scrape design reference images. Falls back to static database if unavailable.
        Returns list of reference dicts with title, description, source, category, pattern_tags.
        """
        if not self._playwright_available:
            return REFERENCE_DESIGNS

        results = []
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                # Scrape Dribbble
                if not sources or "dribbble" in sources:
                    dribbble_refs = await self._scrape_dribbble(page, max_per_source)
                    results.extend(dribbble_refs)

                await browser.close()
        except Exception as e:
            print(f"[Scraper] Playwright scraping failed: {e}. Using static references.")
            return REFERENCE_DESIGNS

        return results if results else REFERENCE_DESIGNS

    async def _scrape_dribbble(self, page, max_items: int = 10) -> list[dict]:
        """Scrape Dribbble design shots for reference patterns."""
        refs = []
        try:
            await page.goto(
                "https://dribbble.com/search/ui-design",
                wait_until="networkidle",
                timeout=15000,
            )
            await page.wait_for_timeout(2000)

            shots = await page.query_selector_all(".shot-thumbnail")
            for shot in shots[:max_items]:
                try:
                    title_el = await shot.query_selector(".shot-title")
                    title = await title_el.inner_text() if title_el else "Untitled"
                    img_el = await shot.query_selector("img")
                    src = await img_el.get_attribute("src") if img_el else None

                    refs.append({
                        "title": title.strip(),
                        "description": f"Dribbble UI design: {title.strip()}",
                        "source": "dribbble",
                        "category": "ui",
                        "pattern_tags": ["dribbble", "ui-design"],
                        "image_url": src,
                    })
                except Exception:
                    continue
        except Exception as e:
            print(f"[Scraper] Dribbble scrape failed: {e}")
        return refs

    def get_static_references(self) -> list[dict]:
        """Return the static reference database directly."""
        return REFERENCE_DESIGNS

    def get_index_stats(self) -> dict:
        """Return stats about the current reference index."""
        refs = REFERENCE_DESIGNS
        cats = {}
        for r in refs:
            c = r.get("category", "unknown")
            cats[c] = cats.get(c, 0) + 1
        return {
            "total": len(refs),
            "playwright_available": self._playwright_available,
            "categories": cats,
            "sources": list({r["source"] for r in refs}),
        }


def build_image_rag_index(references: list[dict], vectorstore) -> list[dict]:
    """
    Add reference design descriptions to the main vector store
    so similarity queries can find relevant reference patterns.
    Returns the enriched reference list.
    """
    from knowledge_base import DESIGN_KNOWLEDGE

    # Append reference design descriptions to vectorstore docs
    extra_docs = [
        {
            "content": f"{ref['title']}: {ref['description']} Tags: {', '.join(ref.get('pattern_tags', []))}",
            "metadata": {
                "category": "competitor",
                "source": ref.get("source", "reference"),
                "severity": "good",
                "ref_category": ref.get("category", "ui"),
            },
        }
        for ref in references
    ]

    # If vectorstore supports dynamic doc addition
    if hasattr(vectorstore, "add_documents"):
        try:
            vectorstore.add_documents(extra_docs)
        except Exception as e:
            print(f"[ImageRAG] Could not add refs to vectorstore: {e}")

    return references
