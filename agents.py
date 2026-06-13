"""
agents.py
LangGraph StateGraph — 6 specialist agents + orchestrator.

Graph topology:
  START → parallel_agents (fan-out 6 agents via ThreadPoolExecutor)
        → orchestrator (synthesize all results)
        → END

Each agent:
  1. Retrieves RAG context (category-filtered from FAISS/TF-IDF)
  2. Builds multimodal message (image + text + RAG context)
  3. Calls free LLM via OpenRouter
  4. Returns parsed JSON result

Uses free vision models: meta-llama/llama-3.2-11b-vision-instruct:free as primary.
"""

from __future__ import annotations
import json
import os
from typing import TypedDict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from rag_engine import retrieve_context
from config import OPENROUTER_BASE, DEFAULT_MODEL, MAX_TOKENS_AGENT, MAX_TOKENS_ORCHESTRATOR


# ── Typed graph state ─────────────────────────────────────────────────────────

class DesignState(TypedDict):
    images: list[dict]          # [{name, b64, media_type, width, height}]
    api_key: str
    selected_model: str
    vectorstore: Any
    # Agent results
    visual_result: Optional[dict]
    ux_result: Optional[dict]
    market_result: Optional[dict]
    accessibility_result: Optional[dict]
    design_system_result: Optional[dict]
    competitor_result: Optional[dict]
    # Synthesis
    final_report: Optional[dict]
    # Runtime
    agent_statuses: dict        # {agent_key: "queued"|"running"|"done"|"error"}
    errors: list[str]


# ── LLM factory ──────────────────────────────────────────────────────────────

def make_llm(api_key: str, model: str, max_tokens: int = MAX_TOKENS_AGENT) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE,
        max_tokens=max_tokens,
        temperature=0.1,
        default_headers={
            "HTTP-Referer": "https://design-analysis-suite.local",
            "X-Title": "Design Analysis Suite",
        },
    )


# ── Shared multimodal caller ──────────────────────────────────────────────────

def _call_llm_vision(
    llm: ChatOpenAI,
    system_prompt: str,
    images: list[dict],
    user_text: str,
    rag_context: str = "",
) -> dict:
    """
    Build and send a multimodal message with 1 or 2 images + RAG context.
    Returns parsed JSON dict from the response.
    """
    full_system = system_prompt
    if rag_context:
        full_system += f"\n\n{rag_context}"

    # Build image content blocks (support up to 2 images for compare mode)
    content_blocks: list = []
    for img in images[:2]:
        content_blocks.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{img['media_type']};base64,{img['b64']}",
                "detail": "high" if len(images) == 1 else "low",
            },
        })

    content_blocks.append({"type": "text", "text": user_text})

    messages = [
        SystemMessage(content=full_system),
        HumanMessage(content=content_blocks),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown fences
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except Exception:
                continue

    return json.loads(raw)


# ── Agent system prompts ──────────────────────────────────────────────────────

VISUAL_SYSTEM = """You are a senior visual designer with 15+ years of product design experience.
Analyze the uploaded UI design screenshot(s) and return ONLY valid JSON, no preamble.

Return exactly this structure:
{
  "score": <integer 0-100>,
  "summary": "<2-sentence visual assessment>",
  "findings": [
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<specific observation referencing actual elements>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<specific observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<specific observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<specific observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<specific observation>"}
  ],
  "top_fix": "<single most impactful visual improvement>",
  "action_items": ["<specific fix 1>", "<specific fix 2>", "<specific fix 3>"]
}

Evaluate: layout hierarchy, typography pairing/scale, color palette harmony, whitespace, visual weight, grid alignment, component consistency.
Be specific — reference actual elements (buttons, headers, cards, colors). Return exactly 5 findings."""

UX_SYSTEM = """You are a UX researcher and interaction designer specializing in usability audits.
Analyze the UI screenshot(s) and return ONLY valid JSON, no preamble.

Return exactly:
{
  "score": <integer 0-100>,
  "summary": "<2-sentence UX assessment>",
  "findings": [
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<usability observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<usability observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<usability observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<usability observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<usability observation>"}
  ],
  "top_fix": "<single most impactful UX improvement>",
  "action_items": ["<fix 1>", "<fix 2>", "<fix 3>"]
}

Evaluate: CTA clarity/placement, cognitive load, information architecture, navigation patterns, Fitts's Law, error prevention, onboarding clarity, loading states.
Return exactly 5 findings."""

MARKET_SYSTEM = """You are a product strategist and competitive design analyst.
Analyze the design screenshot(s) and return ONLY valid JSON, no preamble.

Return exactly:
{
  "score": <integer 0-100>,
  "summary": "<2-sentence market assessment>",
  "findings": [
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<market/competitive observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<market/competitive observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<market/competitive observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<market/competitive observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<market/competitive observation>"}
  ],
  "top_fix": "<single most impactful strategic improvement>",
  "action_items": ["<fix 1>", "<fix 2>", "<fix 3>"],
  "positioning": "<one sentence on market positioning>"
}

Evaluate: design pattern currency (modern vs dated), differentiation signals, target audience fit, industry conventions, design maturity indicators, mobile-first adherence.
Return exactly 5 findings."""

ACCESSIBILITY_SYSTEM = """You are a WCAG 2.1 AA compliance specialist and inclusive design expert.
Analyze the UI screenshot(s) and return ONLY valid JSON, no preamble.

Return exactly:
{
  "score": <integer 0-100>,
  "summary": "<2-sentence accessibility assessment>",
  "findings": [
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<WCAG observation with criterion reference>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<WCAG observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<WCAG observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<WCAG observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<WCAG observation>"}
  ],
  "top_fix": "<single most impactful accessibility fix>",
  "action_items": ["<fix 1>", "<fix 2>", "<fix 3>"],
  "wcag_level": "fails-AA|passes-AA|passes-AAA"
}

Evaluate: contrast ratios (WCAG 1.4.3), touch targets (2.5.5), text legibility, focus indicators, semantic structure, form labels, color-only information, animation concerns.
Return exactly 5 findings."""

DESIGN_SYSTEM_SYSTEM = """You are a design systems architect and component library expert.
Analyze the UI screenshot(s) for design system consistency and return ONLY valid JSON, no preamble.

Return exactly:
{
  "score": <integer 0-100>,
  "summary": "<2-sentence design system assessment>",
  "findings": [
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<design system observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<design system observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<design system observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<design system observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<design system observation>"}
  ],
  "top_fix": "<single most impactful system improvement>",
  "action_items": ["<fix 1>", "<fix 2>", "<fix 3>"],
  "maturity": "ad-hoc|emerging|defined|managed|optimized"
}

Evaluate: spacing token consistency (8px grid), color system coherence, typography scale adherence, border-radius consistency, shadow/elevation system, component naming conventions, icon style consistency.
Return exactly 5 findings."""

COMPETITOR_SYSTEM = """You are a competitive intelligence analyst specializing in UI/UX benchmarking.
Analyze the design screenshot(s) against industry reference patterns and return ONLY valid JSON, no preamble.

Return exactly:
{
  "score": <integer 0-100>,
  "summary": "<2-sentence competitive analysis>",
  "findings": [
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<competitive observation vs industry patterns>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<competitive observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<competitive observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<competitive observation>"},
    {"severity": "critical|warning|good", "title": "<short title>", "detail": "<competitive observation>"}
  ],
  "top_fix": "<single most impactful competitive improvement>",
  "action_items": ["<fix 1>", "<fix 2>", "<fix 3>"],
  "similar_patterns": ["<reference pattern 1>", "<reference pattern 2>"],
  "differentiation_score": <integer 0-100>
}

Evaluate: similarity to Dribbble/Mobbin reference patterns, industry convention adherence, innovation vs convention balance, trend adoption (bento grid, glassmorphism, etc.), dark mode support signals.
Return exactly 5 findings."""

ORCHESTRATOR_SYSTEM = """You are a design director synthesizing a multi-agent design audit into an executive report.
You receive JSON from 6 specialist agents. Return ONLY valid JSON, no preamble.

Return exactly:
{
  "overall_score": <integer 0-100>,
  "verdict": "<one punchy sentence on the design's current state>",
  "design_maturity": "early|developing|polished|production-ready",
  "score_breakdown": {
    "visual": <score>, "ux": <score>, "market": <score>,
    "accessibility": <score>, "design_system": <score>, "competitor": <score>
  },
  "priority_fixes": [
    {"rank": 1, "area": "<agent>", "action": "<specific fix>", "impact": "high|medium|low", "effort": "low|medium|high"},
    {"rank": 2, "area": "<agent>", "action": "<specific fix>", "impact": "high|medium|low", "effort": "low|medium|high"},
    {"rank": 3, "area": "<agent>", "action": "<specific fix>", "impact": "high|medium|low", "effort": "low|medium|high"},
    {"rank": 4, "area": "<agent>", "action": "<specific fix>", "impact": "high|medium|low", "effort": "low|medium|high"},
    {"rank": 5, "area": "<agent>", "action": "<specific fix>", "impact": "high|medium|low", "effort": "low|medium|high"}
  ],
  "quick_wins": ["<fix doable in <1 hour>", "<fix doable in <1 hour>", "<fix doable in <1 hour>"],
  "strengths": ["<genuine strength>", "<strength>", "<strength>"],
  "total_action_items": <integer count of all action items across agents>
}"""


# ── Individual agent node functions ───────────────────────────────────────────

def _run_single_agent(
    key: str,
    system: str,
    user_text: str,
    rag_query: str,
    rag_category: str,
    state: DesignState,
) -> dict:
    """Generic agent runner — fetch RAG, call LLM, parse JSON."""
    try:
        llm = make_llm(state["api_key"], state["selected_model"])
        rag_ctx = retrieve_context(state["vectorstore"], rag_query, rag_category, k=3)
        result = _call_llm_vision(llm, system, state["images"], user_text, rag_ctx)
        return {f"{key}_result": result}
    except Exception as e:
        return {
            f"{key}_result": None,
            "errors": state.get("errors", []) + [f"{key}: {str(e)[:200]}"],
        }


def parallel_agents_node(state: DesignState) -> dict:
    """Fan-out: run all 6 agents in parallel via ThreadPoolExecutor."""
    agent_specs = {
        "visual": (
            VISUAL_SYSTEM,
            "Analyze the visual design of this screenshot.",
            "visual design layout typography color hierarchy",
            "visual",
        ),
        "ux": (
            UX_SYSTEM,
            "Analyze the UX and usability of this design.",
            "UX usability CTA navigation cognitive load friction",
            "ux",
        ),
        "market": (
            MARKET_SYSTEM,
            "Analyze the market positioning and competitive context of this design.",
            "market competitive patterns industry conventions mobile",
            "market",
        ),
        "accessibility": (
            ACCESSIBILITY_SYSTEM,
            "Perform a WCAG 2.1 accessibility audit on this design.",
            "WCAG accessibility contrast keyboard screen reader",
            "accessibility",
        ),
        "design_system": (
            DESIGN_SYSTEM_SYSTEM,
            "Audit this design for design system consistency and token usage.",
            "design system tokens spacing consistency components",
            "design_system",
        ),
        "competitor": (
            COMPETITOR_SYSTEM,
            "Compare this design against industry reference patterns from Dribbble and Mobbin.",
            "competitor reference patterns Dribbble Mobbin industry benchmark",
            "competitor",
        ),
    }

    results = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(_run_single_agent, key, spec[0], spec[1], spec[2], spec[3], state): key
            for key, spec in agent_specs.items()
        }
        for future in as_completed(futures):
            results.update(future.result())

    return results


def orchestrator_node(state: DesignState) -> dict:
    """Synthesize all 6 agent results into a final priority-ranked report."""
    try:
        llm = make_llm(state["api_key"], state["selected_model"], MAX_TOKENS_ORCHESTRATOR)

        agent_data = {
            "visual": state.get("visual_result"),
            "ux": state.get("ux_result"),
            "market": state.get("market_result"),
            "accessibility": state.get("accessibility_result"),
            "design_system": state.get("design_system_result"),
            "competitor": state.get("competitor_result"),
        }

        user_text = (
            f"Here are the 6 specialist agent reports:\n\n"
            f"{json.dumps(agent_data, indent=2)}\n\n"
            f"Synthesize these into the final orchestrator report."
        )

        # Use low detail for orchestrator (save tokens — it already has text summaries)
        orch_images = [{**img, "b64": img["b64"][:100] + "..."} for img in state["images"][:1]]

        result = _call_llm_vision(
            llm, ORCHESTRATOR_SYSTEM, state["images"][:1], user_text, ""
        )
        return {"final_report": result}
    except Exception as e:
        # Build a fallback report from available agent scores
        scores = []
        breakdown = {}
        for key in ["visual", "ux", "market", "accessibility", "design_system", "competitor"]:
            r = state.get(f"{key}_result")
            s = r.get("score", 50) if r else 50
            scores.append(s)
            breakdown[key] = s

        overall = int(sum(scores) / len(scores)) if scores else 50
        return {
            "final_report": {
                "overall_score": overall,
                "verdict": "Analysis complete — see individual agent reports for details.",
                "design_maturity": "developing",
                "score_breakdown": breakdown,
                "priority_fixes": [],
                "quick_wins": [],
                "strengths": [],
                "total_action_items": 0,
            },
            "errors": state.get("errors", []) + [f"Orchestrator: {str(e)[:200]}"],
        }


# ── Build compiled graph ──────────────────────────────────────────────────────

def build_graph() -> Any:
    graph = StateGraph(DesignState)
    graph.add_node("parallel_agents", parallel_agents_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.set_entry_point("parallel_agents")
    graph.add_edge("parallel_agents", "orchestrator")
    graph.add_edge("orchestrator", END)
    return graph.compile()


compiled_graph = build_graph()
