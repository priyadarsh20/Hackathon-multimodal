"""
config.py — central config for the Design Analysis Suite.
Change FREE_MODELS list to swap LLMs. All use OpenRouter.
"""

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# ── Free vision-capable models on OpenRouter (low token usage) ────────────────
# Ordered by preference: fastest/cheapest first
FREE_VISION_MODELS = [
    "meta-llama/llama-3.2-11b-vision-instruct:free",   # best free vision model
    "google/gemini-2.0-flash-exp:free",                 # Gemini free tier
    "qwen/qwen2-vl-7b-instruct:free",                   # Qwen vision, very fast
    "microsoft/phi-3.5-vision-instruct:free",           # small, efficient
]

# Default — change to any model in FREE_VISION_MODELS
DEFAULT_MODEL = FREE_VISION_MODELS[0]

# Token budget per agent (keeps costs at zero on free tier)
MAX_TOKENS_AGENT = 800
MAX_TOKENS_ORCHESTRATOR = 1000

# ── Agent roster (matches the UI screenshot) ──────────────────────────────────
AGENTS = {
    "visual": {
        "name": "Visual analysis",
        "subtitle": "Layout, color, typography, hierarchy",
        "emoji": "👁",
        "color": "#7c3aed",
        "bg": "#f5f3ff",
    },
    "ux": {
        "name": "UX critique",
        "subtitle": "Flows, friction, cognitive load",
        "emoji": "🧭",
        "color": "#0369a1",
        "bg": "#f0f9ff",
    },
    "market": {
        "name": "Market research",
        "subtitle": "Competitor benchmarking, positioning",
        "emoji": "📊",
        "color": "#b45309",
        "bg": "#fffbeb",
    },
    "accessibility": {
        "name": "Accessibility audit",
        "subtitle": "WCAG 2.1, contrast, touch targets",
        "emoji": "♿",
        "color": "#0f766e",
        "bg": "#f0fdfa",
    },
    "design_system": {
        "name": "Design system",
        "subtitle": "Token consistency, component audit",
        "emoji": "🧩",
        "color": "#be185d",
        "bg": "#fdf2f8",
    },
    "competitor": {
        "name": "Competitor compare",
        "subtitle": "RAG similarity, market positioning",
        "emoji": "⚡",
        "color": "#1d4ed8",
        "bg": "#eff6ff",
    },
}

AGENT_ORDER = ["visual", "ux", "market", "accessibility", "design_system", "competitor"]

# ── RAG knowledge base categories ────────────────────────────────────────────
RAG_CATEGORIES = ["visual", "ux", "accessibility", "market", "design_system", "competitor"]

# ── Playwright scrape targets for Image RAG ───────────────────────────────────
SCRAPE_SOURCES = [
    {"name": "Dribbble", "url": "https://dribbble.com/search/ui-design", "selector": ".shot-thumbnail img"},
    {"name": "Mobbin",   "url": "https://mobbin.com/browse/ios/apps",    "selector": "img.screenshot"},
]
