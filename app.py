"""
app.py — Design Analysis Suite · Streamlit UI
Matches the design from the reference screenshot:
  - Left panel: upload + agent roster + RAG index stats + Run button
  - Right panel: tabs (Overview · Agent outputs · Compare · Image RAG · Report)
  - Live agent status: Queued → Running → Done
  - 6 agents: Visual · UX · Market · Accessibility · Design System · Competitor
"""

import streamlit as st
import base64
import json
import os
import time
from pathlib import Path
from datetime import datetime
from PIL import Image
import io
from dotenv import load_dotenv

from config import AGENTS, AGENT_ORDER, FREE_VISION_MODELS, DEFAULT_MODEL
from rag_engine import build_vectorstore
from agents import compiled_graph, DesignState
from playwright_scraper import DesignScraper, REFERENCE_DESIGNS

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Design Analysis Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS — matches the reference screenshot aesthetic ──────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f8f7f4; }
  .main .block-container { padding: 1rem 1.5rem; max-width: 1300px; }
  footer, header { display: none !important; }

  /* Header badge chips */
  .badge-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0 16px; }
  .badge { display: inline-flex; align-items: center; gap: 5px; padding: 4px 12px;
           border-radius: 999px; font-size: 12px; font-weight: 500; border: 1px solid; }
  .badge-purple { background: #f5f3ff; color: #6d28d9; border-color: #c4b5fd; }
  .badge-green  { background: #f0fdf4; color: #166534; border-color: #86efac; }
  .badge-blue   { background: #eff6ff; color: #1d4ed8; border-color: #93c5fd; }
  .badge-amber  { background: #fffbeb; color: #92400e; border-color: #fcd34d; }
  .badge-teal   { background: #f0fdfa; color: #0f766e; border-color: #5eead4; }

  /* Upload zone */
  .upload-zone { border: 2px dashed #d1d5db; border-radius: 12px; padding: 28px;
                 text-align: center; background: #fafafa; cursor: pointer; }
  .upload-zone:hover { border-color: #7c3aed; background: #faf5ff; }

  /* Agent roster row */
  .agent-row { display: flex; align-items: center; gap: 10px; padding: 10px 12px;
               border-radius: 8px; margin: 4px 0; font-size: 14px; cursor: pointer; }
  .agent-row.active { background: #f5f3ff; }
  .agent-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .dot-queued  { background: #d1d5db; }
  .dot-running { background: #f59e0b; animation: pulse 1s infinite; }
  .dot-done    { background: #22c55e; }
  .dot-error   { background: #ef4444; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* Status row for agent outputs tab */
  .status-row { display: flex; align-items: center; padding: 14px 16px;
                border: 1px solid #e5e7eb; border-radius: 10px; margin: 6px 0;
                background: #fff; gap: 12px; }
  .status-row.done    { border-color: #bbf7d0; background: #f0fdf4; }
  .status-row.running { border-color: #fde68a; background: #fffbeb; }
  .status-row.error   { border-color: #fecaca; background: #fef2f2; }

  /* Score display */
  .score-giant { font-size: 52px; font-weight: 800; line-height: 1; }
  .score-label { font-size: 12px; color: #9ca3af; text-transform: uppercase; letter-spacing: .05em; }

  /* Metric card */
  .metric-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px;
                 padding: 20px; text-align: center; }
  .metric-num  { font-size: 36px; font-weight: 700; }
  .metric-lbl  { font-size: 13px; color: #6b7280; margin-top: 2px; }

  /* Finding blocks */
  .finding-critical { background:#fef2f2; border-left:3px solid #fca5a5; border-radius:0 8px 8px 0; padding:10px 14px; margin:5px 0; }
  .finding-warning  { background:#fffbeb; border-left:3px solid #fcd34d; border-radius:0 8px 8px 0; padding:10px 14px; margin:5px 0; }
  .finding-good     { background:#f0fdf4; border-left:3px solid #86efac; border-radius:0 8px 8px 0; padding:10px 14px; margin:5px 0; }

  /* Fix box */
  .fix-box { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:10px 14px; margin-top:10px; }

  /* Priority fix row */
  .priority-row { display:flex; align-items:flex-start; gap:12px; padding:12px 0;
                  border-bottom:1px solid #f3f4f6; }
  .rank-num { font-size:24px; font-weight:800; color:#e5e7eb; min-width:32px; }

  /* Tab styling */
  [data-testid="stTabs"] [data-baseweb="tab"] { font-size:14px; }
  [data-testid="stTabs"] [aria-selected="true"] { color: #7c3aed !important; }

  /* Image RAG card */
  .rag-ref-card { background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:14px; margin:6px 0; }

  /* Run button override */
  div[data-testid="stButton"] button[kind="primary"] {
    background: #111827; color: #fff; border: none;
    border-radius: 8px; font-size: 15px; font-weight: 500;
    padding: 12px; width: 100%;
  }
  div[data-testid="stButton"] button[kind="primary"]:hover { background: #374151; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "analysis_done": False,
        "agent_statuses": {k: "queued" for k in AGENT_ORDER},
        "agent_results": {},
        "final_report": None,
        "uploaded_images": [],
        "errors": [],
        "analyzed_at": None,
        "rag_index_stats": {"total": len(REFERENCE_DESIGNS), "playwright_available": False, "categories": {}},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Helper functions ──────────────────────────────────────────────────────────

def score_color(s: int) -> str:
    return "#16a34a" if s >= 75 else "#d97706" if s >= 50 else "#dc2626"

def encode_image(file) -> dict:
    suffix = Path(file.name).suffix.lower()
    media_map = {".jpg":"image/jpeg",".jpeg":"image/jpeg",".png":"image/png",".webp":"image/webp"}
    media_type = media_map.get(suffix, "image/png")
    data = file.read()
    img = Image.open(io.BytesIO(data))
    w, h = img.size
    b64 = base64.standard_b64encode(data).decode()
    return {"name": file.name, "b64": b64, "media_type": media_type, "width": w, "height": h, "data": data}

def render_findings(findings: list):
    for f in findings:
        sev = f.get("severity", "warning")
        icon = "🔴" if sev=="critical" else "🟡" if sev=="warning" else "🟢"
        color = "#991b1b" if sev=="critical" else "#92400e" if sev=="warning" else "#166534"
        st.markdown(
            f'<div class="finding-{sev}">'
            f'<div style="font-size:13px;font-weight:600;color:{color}">{icon} {f.get("title","")}</div>'
            f'<div style="font-size:12px;color:{color};margin-top:3px;opacity:.85">{f.get("detail","")}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

def status_icon(status: str) -> str:
    return {"queued":"⏳","running":"🔄","done":"✅","error":"❌"}.get(status, "⏳")

def status_label(status: str) -> str:
    return {"queued":"Queued","running":"Running","done":"Done","error":"Error"}.get(status, "Queued")

@st.cache_resource(show_spinner="📚 Building knowledge base…")
def get_vectorstore(hf_token: str):
    return build_vectorstore(hf_token)


# ── HEADER ────────────────────────────────────────────────────────────────────
col_icon, col_title = st.columns([0.05, 0.95])
with col_icon:
    st.markdown('<div style="font-size:36px;margin-top:4px">🔬</div>', unsafe_allow_html=True)
with col_title:
    st.markdown('<h2 style="margin:0;font-size:22px;font-weight:700">Design Analysis Suite</h2>', unsafe_allow_html=True)
    st.markdown('<p style="margin:0;color:#6b7280;font-size:14px">Multimodal AI agents for product & app design</p>', unsafe_allow_html=True)

n_agents = len(AGENT_ORDER)
rag_count = st.session_state["rag_index_stats"]["total"]
st.markdown(f"""
<div class="badge-row">
  <span class="badge badge-purple">🤖 {n_agents} agents ready</span>
  <span class="badge badge-green">🖼 Image RAG enabled</span>
  <span class="badge badge-blue">🧠 Multimodal LLM</span>
  <span class="badge badge-amber">📐 Design benchmarks</span>
  <span class="badge badge-teal">🕸 Agent mesh</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── TWO-COLUMN LAYOUT ─────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2.5], gap="large")


# ════════════════════════════════════════════════════════════════════════════
# LEFT PANEL
# ════════════════════════════════════════════════════════════════════════════
with left_col:

    # ── Settings expander ────────────────────────────────────────────────
    with st.expander("⚙ Settings", expanded=False):
        api_key = st.text_input(
            "OpenRouter API key",
            value=os.environ.get("OPENROUTER_API_KEY", ""),
            type="password",
            placeholder="sk-or-v1-...",
        )
        hf_token = st.text_input(
            "HuggingFace token (optional — for cloud embeddings)",
            value=os.environ.get("HF_TOKEN", ""),
            type="password",
            placeholder="hf_...",
            help="Without this, TF-IDF local embeddings are used for RAG",
        )
        selected_model = st.selectbox(
            "LLM model (all free on OpenRouter)",
            FREE_VISION_MODELS,
            index=0,
        )
        st.caption(f"Model: `{selected_model}`")

    st.markdown("---")

    # ── Upload ───────────────────────────────────────────────────────────
    st.markdown('<div style="font-size:28px;text-align:center;margin:8px 0">🖼</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;font-weight:600;margin:4px 0">Upload designs</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;font-size:12px;color:#6b7280">PNG, JPG, Figma export · up to 10 files</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload designs",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        imgs = []
        for f in uploaded_files[:10]:
            f.seek(0)
            imgs.append(encode_image(f))
        st.session_state["uploaded_images"] = imgs

        for img in imgs:
            st.caption(f"📄 **{img['name']}** · {img['width']}×{img['height']}")

    st.markdown("---")

    # ── Agent roster ─────────────────────────────────────────────────────
    st.markdown('<p style="font-size:11px;font-weight:600;color:#9ca3af;letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px">AGENT ROSTER</p>', unsafe_allow_html=True)

    statuses = st.session_state["agent_statuses"]
    for i, key in enumerate(AGENT_ORDER):
        cfg = AGENTS[key]
        status = statuses.get(key, "queued")
        dot_class = f"dot-{status}"
        is_active = (i == 0)
        row_class = "agent-row active" if is_active else "agent-row"
        status_badge = ""
        if status == "running":
            status_badge = f'<span style="margin-left:auto;font-size:11px;background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:999px;font-weight:600">Running</span>'
        elif status == "done":
            status_badge = f'<span style="margin-left:auto;font-size:11px;background:#dcfce7;color:#166534;padding:2px 8px;border-radius:999px;font-weight:600">Done</span>'

        st.markdown(
            f'<div class="{row_class}">'
            f'  <div class="agent-dot {dot_class}"></div>'
            f'  <span style="font-size:16px">{cfg["emoji"]}</span>'
            f'  <span style="font-size:14px;font-weight:{"600" if is_active else "400"}">{cfg["name"]}</span>'
            f'  {status_badge}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── RAG Image Index ───────────────────────────────────────────────────
    st.markdown('<p style="font-size:11px;font-weight:600;color:#9ca3af;letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px">RAG IMAGE INDEX</p>', unsafe_allow_html=True)
    stats = st.session_state["rag_index_stats"]
    st.markdown(
        f'<p style="font-size:13px;color:#374151;margin:0">'
        f'<strong>{stats["total"]}</strong> reference designs indexed.<br>'
        f'Similarity search active across Dribbble, Mobbin, and uploaded history.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Run button ────────────────────────────────────────────────────────
    has_images = len(st.session_state["uploaded_images"]) > 0
    has_key = bool(api_key and api_key.strip())

    run_clicked = st.button(
        "▶  Run all agents",
        type="primary",
        use_container_width=True,
        disabled=not (has_images and has_key),
    )
    if not has_key:
        st.caption("⚠ Add OpenRouter key in Settings ↑")
    if not has_images:
        st.caption("⚠ Upload at least one design image ↑")


# ════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — TABS
# ════════════════════════════════════════════════════════════════════════════
with right_col:
    tab_overview, tab_agents, tab_compare, tab_rag, tab_report = st.tabs([
        "Overview", "Agent outputs", "Compare", "Image RAG", "Report"
    ])


    # ── RUN PIPELINE (triggered from left panel) ──────────────────────────
    if run_clicked and has_images and has_key:
        # Reset state
        st.session_state["analysis_done"] = False
        st.session_state["agent_statuses"] = {k: "queued" for k in AGENT_ORDER}
        st.session_state["agent_results"] = {}
        st.session_state["final_report"] = None
        st.session_state["errors"] = []

        with tab_overview:
            progress = st.progress(0, text="🚀 Initializing agents…")
            status_box = st.empty()

        # Build vectorstore
        vectorstore = get_vectorstore(hf_token.strip() if hf_token else "")

        # Mark all as running
        st.session_state["agent_statuses"] = {k: "running" for k in AGENT_ORDER}

        with tab_overview:
            progress.progress(15, text="🔄 Running 6 agents in parallel…")
            status_box.info("Agents are analyzing your design simultaneously via LangGraph…")

        # Build initial state
        initial_state: DesignState = {
            "images": st.session_state["uploaded_images"],
            "api_key": api_key.strip(),
            "selected_model": selected_model,
            "vectorstore": vectorstore,
            "visual_result": None,
            "ux_result": None,
            "market_result": None,
            "accessibility_result": None,
            "design_system_result": None,
            "competitor_result": None,
            "final_report": None,
            "agent_statuses": {},
            "errors": [],
        }

        try:
            result = compiled_graph.invoke(initial_state)

            # Store results
            agent_results = {
                "visual": result.get("visual_result"),
                "ux": result.get("ux_result"),
                "market": result.get("market_result"),
                "accessibility": result.get("accessibility_result"),
                "design_system": result.get("design_system_result"),
                "competitor": result.get("competitor_result"),
            }

            statuses = {}
            for k in AGENT_ORDER:
                statuses[k] = "done" if agent_results.get(k) else "error"

            st.session_state["agent_statuses"] = statuses
            st.session_state["agent_results"] = agent_results
            st.session_state["final_report"] = result.get("final_report")
            st.session_state["analysis_done"] = True
            st.session_state["analyzed_at"] = datetime.now().isoformat()
            st.session_state["errors"] = result.get("errors", [])

            with tab_overview:
                progress.progress(100, text="✅ Analysis complete!")
                status_box.success("All agents done! See tabs for detailed results.")

        except Exception as e:
            st.session_state["agent_statuses"] = {k: "error" for k in AGENT_ORDER}
            st.session_state["errors"] = [str(e)]
            with tab_overview:
                progress.empty()
                status_box.error(f"Pipeline error: {e}")

        st.rerun()


    # ════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ════════════════════════════════════════════════════════════════════
    with tab_overview:
        if not st.session_state["analysis_done"]:
            # Pre-run state — show uploaded image preview
            imgs = st.session_state["uploaded_images"]
            if imgs:
                cols = st.columns(min(len(imgs), 3))
                for i, img in enumerate(imgs[:3]):
                    with cols[i]:
                        label = "Design A · home screen" if i == 0 else f"Design {chr(65+i)}"
                        st.image(img["data"], caption=label, use_container_width=True)
                        st.caption(f"**{img['name']}** · {img['width']}×{img['height']}")
                st.info("👈 Configure your settings and click **Run all agents** to start analysis.")
            else:
                st.markdown("""
<div style="text-align:center;padding:60px 20px;color:#9ca3af">
  <div style="font-size:48px;margin-bottom:12px">🖼</div>
  <div style="font-size:16px;font-weight:500;margin-bottom:6px">Upload a design to get started</div>
  <div style="font-size:13px">PNG, JPG, or Figma export · Use the panel on the left</div>
</div>""", unsafe_allow_html=True)
        else:
            report = st.session_state["final_report"] or {}
            agent_results = st.session_state["agent_results"]

            # ── Metric cards row ──
            m1, m2, m3 = st.columns(3)
            overall = report.get("overall_score", 0)
            agents_run = sum(1 for v in agent_results.values() if v)
            action_items = sum(
                len(v.get("action_items", []))
                for v in agent_results.values() if v
            )
            with m1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-num" style="color:{score_color(overall)}">{overall}</div>'
                    f'<div class="metric-lbl">Overall score</div></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-num" style="color:#7c3aed">{agents_run}</div>'
                    f'<div class="metric-lbl">Agents run</div></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-num" style="color:#0369a1">{action_items}</div>'
                    f'<div class="metric-lbl">Action items</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div style="margin:16px 0"></div>', unsafe_allow_html=True)

            # ── Agent status rows ──
            statuses = st.session_state["agent_statuses"]
            for key in AGENT_ORDER:
                cfg = AGENTS[key]
                status = statuses.get(key, "queued")
                icon = status_icon(status)
                lbl = status_label(status)
                badge_style = (
                    "background:#dcfce7;color:#166534" if status=="done" else
                    "background:#fef3c7;color:#92400e" if status=="running" else
                    "background:#f3f4f6;color:#374151"
                )
                st.markdown(
                    f'<div class="status-row {status}">'
                    f'  <span style="font-size:18px">{icon}</span>'
                    f'  <div style="flex:1">'
                    f'    <div style="font-weight:600;font-size:14px">{cfg["emoji"]} {cfg["name"]} agent</div>'
                    f'    <div style="font-size:12px;color:#6b7280">{cfg["subtitle"]}</div>'
                    f'  </div>'
                    f'  <span style="font-size:12px;font-weight:600;padding:3px 12px;border-radius:999px;{badge_style}">{lbl}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # ── Verdict ──
            verdict = report.get("verdict", "")
            if verdict:
                st.markdown(f'<div style="margin-top:16px;padding:14px 18px;background:#faf5ff;border:1px solid #e9d5ff;border-radius:10px;font-size:14px;color:#6d28d9"><strong>🧠 Verdict:</strong> {verdict}</div>', unsafe_allow_html=True)

            # ── Quick wins ──
            quick_wins = report.get("quick_wins", [])
            if quick_wins:
                st.markdown("**⚡ Quick wins (< 1 hour)**")
                for w in quick_wins:
                    st.markdown(f'<div style="font-size:13px;padding:4px 0;color:#166534">✓ {w}</div>', unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════
    # TAB 2 — AGENT OUTPUTS
    # ════════════════════════════════════════════════════════════════════
    with tab_agents:
        if not st.session_state["analysis_done"]:
            st.info("Run analysis first to see individual agent outputs.")
        else:
            agent_results = st.session_state["agent_results"]
            selected_agent = st.radio(
                "Select agent",
                AGENT_ORDER,
                format_func=lambda k: f"{AGENTS[k]['emoji']} {AGENTS[k]['name']}",
                horizontal=True,
                label_visibility="collapsed",
            )
            st.markdown("---")

            data = agent_results.get(selected_agent)
            cfg = AGENTS[selected_agent]

            if data is None:
                st.error(f"{cfg['emoji']} {cfg['name']} agent returned no data. Check errors below.")
            else:
                # Score + summary
                col_info, col_score = st.columns([3, 1])
                with col_info:
                    st.markdown(f'<h3 style="margin:0;color:{cfg["color"]}">{cfg["emoji"]} {cfg["name"]}</h3>', unsafe_allow_html=True)
                    st.markdown(f'<p style="color:#6b7280;font-size:14px;margin:6px 0 0">{data.get("summary","")}</p>', unsafe_allow_html=True)

                    # Extra fields per agent
                    if selected_agent == "accessibility":
                        wcag = data.get("wcag_level", "")
                        if wcag:
                            wc = "#16a34a" if "passes" in wcag else "#dc2626"
                            st.markdown(f'<span style="font-size:12px;font-weight:600;color:{wc};background:{wc}18;padding:3px 10px;border-radius:999px">{wcag.upper()}</span>', unsafe_allow_html=True)
                    elif selected_agent == "design_system":
                        mat = data.get("maturity", "")
                        if mat:
                            st.markdown(f'<span style="font-size:12px;font-weight:600;color:#7c3aed;background:#f5f3ff;padding:3px 10px;border-radius:999px">System maturity: {mat}</span>', unsafe_allow_html=True)
                    elif selected_agent == "competitor":
                        diff = data.get("differentiation_score")
                        if diff is not None:
                            st.markdown(f'<span style="font-size:12px;font-weight:600;color:#1d4ed8;background:#eff6ff;padding:3px 10px;border-radius:999px">Differentiation score: {diff}/100</span>', unsafe_allow_html=True)
                        patterns = data.get("similar_patterns", [])
                        if patterns:
                            st.caption(f"Similar patterns: {' · '.join(patterns)}")
                    elif selected_agent == "market":
                        pos = data.get("positioning", "")
                        if pos:
                            st.caption(f"📍 {pos}")

                with col_score:
                    sc = data.get("score", 0)
                    st.markdown(
                        f'<div style="text-align:right">'
                        f'<div class="score-giant" style="color:{score_color(sc)}">{sc}</div>'
                        f'<div class="score-label">/ 100</div></div>',
                        unsafe_allow_html=True,
                    )

                st.progress(data.get("score", 0) / 100)
                st.markdown("**Findings**")
                render_findings(data.get("findings", []))

                top_fix = data.get("top_fix", "")
                if top_fix:
                    st.markdown(
                        f'<div class="fix-box"><strong style="color:#166534">💡 Top fix:</strong> '
                        f'<span style="color:#166534;font-size:13px">{top_fix}</span></div>',
                        unsafe_allow_html=True,
                    )

                action_items = data.get("action_items", [])
                if action_items:
                    st.markdown("**Action items**")
                    for item in action_items:
                        st.markdown(f"- {item}")

            # Errors
            if st.session_state["errors"]:
                with st.expander("⚠ Errors / warnings"):
                    for err in st.session_state["errors"]:
                        st.warning(err)


    # ════════════════════════════════════════════════════════════════════
    # TAB 3 — COMPARE
    # ════════════════════════════════════════════════════════════════════
    with tab_compare:
        imgs = st.session_state["uploaded_images"]
        if len(imgs) < 2:
            st.info("Upload 2+ design images to use Compare mode. Both designs will be analyzed side by side.")
            if imgs:
                st.image(imgs[0]["data"], caption=imgs[0]["name"], width=400)
        else:
            st.markdown("### Side-by-side comparison")
            img_a, img_b = imgs[0], imgs[1]

            col_a, col_b = st.columns(2)
            with col_a:
                st.image(img_a["data"], caption=f"Design A · {img_a['name']}", use_container_width=True)
                st.caption(f"{img_a['width']}×{img_a['height']}")
            with col_b:
                st.image(img_b["data"], caption=f"Design B · {img_b['name']}", use_container_width=True)
                st.caption(f"{img_b['width']}×{img_b['height']}")

            if st.session_state["analysis_done"]:
                st.markdown("---")
                st.markdown("### Score comparison")
                report = st.session_state["final_report"] or {}
                breakdown = report.get("score_breakdown", {})

                if breakdown:
                    import json as _json
                    # Build comparison bars
                    for key in AGENT_ORDER:
                        cfg = AGENTS[key]
                        score = breakdown.get(key, 0)
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0">'
                            f'  <span style="font-size:13px;min-width:140px">{cfg["emoji"]} {cfg["name"]}</span>'
                            f'  <div style="flex:1;background:#e5e7eb;height:8px;border-radius:4px">'
                            f'    <div style="background:{cfg["color"]};width:{score}%;height:8px;border-radius:4px"></div>'
                            f'  </div>'
                            f'  <span style="font-size:13px;font-weight:600;color:{score_color(score)};min-width:32px">{score}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.info("Run analysis to see score comparison between designs.")


    # ════════════════════════════════════════════════════════════════════
    # TAB 4 — IMAGE RAG
    # ════════════════════════════════════════════════════════════════════
    with tab_rag:
        st.markdown("### Image RAG Index")
        st.caption("Reference designs indexed for similarity search. Playwright scrapes Dribbble & Mobbin when available; static database used as fallback.")

        stats = st.session_state["rag_index_stats"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Designs indexed", stats["total"])
        with c2:
            pw = "✅ Active" if stats.get("playwright_available") else "⚠ Fallback mode"
            st.metric("Playwright", pw)
        with c3:
            st.metric("Sources", "Dribbble · Mobbin · Uploads")

        st.markdown("---")
        st.markdown("**Reference design patterns**")

        # Search
        search_q = st.text_input("Search reference patterns", placeholder="e.g. dashboard, checkout, onboarding…")

        filtered = REFERENCE_DESIGNS
        if search_q:
            q_lower = search_q.lower()
            filtered = [
                r for r in REFERENCE_DESIGNS
                if q_lower in r["title"].lower()
                or q_lower in r["description"].lower()
                or any(q_lower in t for t in r.get("pattern_tags", []))
            ]

        # Category filter
        all_cats = sorted(set(r["category"] for r in REFERENCE_DESIGNS))
        cat_filter = st.multiselect("Filter by category", all_cats, default=[])
        if cat_filter:
            filtered = [r for r in filtered if r["category"] in cat_filter]

        st.caption(f"Showing {len(filtered)} of {len(REFERENCE_DESIGNS)} reference designs")

        for ref in filtered:
            tags_html = " ".join(
                f'<span style="font-size:11px;background:#f3f4f6;color:#374151;padding:2px 8px;border-radius:999px">{t}</span>'
                for t in ref.get("pattern_tags", [])
            )
            st.markdown(
                f'<div class="rag-ref-card">'
                f'  <div style="font-weight:600;font-size:14px;margin-bottom:4px">📱 {ref["title"]}</div>'
                f'  <div style="font-size:13px;color:#374151;margin-bottom:8px">{ref["description"]}</div>'
                f'  <div style="display:flex;gap:4px;flex-wrap:wrap">{tags_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


    # ════════════════════════════════════════════════════════════════════
    # TAB 5 — REPORT (export)
    # ════════════════════════════════════════════════════════════════════
    with tab_report:
        if not st.session_state["analysis_done"]:
            st.info("Run analysis to generate the full report.")
        else:
            report = st.session_state["final_report"] or {}
            agent_results = st.session_state["agent_results"]
            overall = report.get("overall_score", 0)
            maturity = report.get("design_maturity", "developing")

            # ── Orchestrator summary ──
            st.markdown(f'<h3 style="color:#7c3aed;margin:0">🧠 Orchestrator Report</h3>', unsafe_allow_html=True)
            verdict = report.get("verdict", "")
            st.markdown(f'<p style="font-size:14px;font-style:italic;color:#374151;margin:6px 0 16px">"{verdict}"</p>', unsafe_allow_html=True)

            cr, sr = st.columns([2, 1])
            with cr:
                mat_icon = {"early":"🌱","developing":"🔧","polished":"✨","production-ready":"🚀"}.get(maturity,"🔧")
                st.markdown(f'<span style="font-size:13px;background:#ede9fe;color:#6d28d9;padding:4px 14px;border-radius:999px;font-weight:600">{mat_icon} {maturity.title()}</span>', unsafe_allow_html=True)
            with sr:
                st.markdown(
                    f'<div style="text-align:right">'
                    f'<span class="score-giant" style="color:{score_color(overall)}">{overall}</span>'
                    f'<span class="score-label"> / 100</span></div>',
                    unsafe_allow_html=True,
                )

            st.progress(overall / 100)

            # ── Score breakdown ──
            st.markdown("**Score breakdown**")
            breakdown = report.get("score_breakdown", {})
            cols = st.columns(3)
            for i, key in enumerate(AGENT_ORDER):
                cfg = AGENTS[key]
                sc = breakdown.get(key, agent_results.get(key, {}) and agent_results[key].get("score", 0) or 0)
                with cols[i % 3]:
                    st.markdown(
                        f'<div style="text-align:center;padding:10px;background:{cfg["bg"]};border-radius:8px;margin:4px 0">'
                        f'  <div style="font-size:22px;font-weight:700;color:{cfg["color"]}">{sc}</div>'
                        f'  <div style="font-size:12px;color:#6b7280">{cfg["emoji"]} {cfg["name"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            # ── Priority fixes ──
            st.markdown("**Priority fixes**")
            IMPACT_COLOR = {"high":"#dc2626","medium":"#d97706","low":"#16a34a"}
            EFFORT_COLOR = {"low":"#16a34a","medium":"#d97706","high":"#dc2626"}

            for fix in report.get("priority_fixes", []):
                imp = fix.get("impact","medium")
                eff = fix.get("effort","medium")
                st.markdown(
                    f'<div class="priority-row">'
                    f'  <span class="rank-num">#{fix.get("rank","")}</span>'
                    f'  <div style="flex:1">'
                    f'    <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;font-weight:600">{fix.get("area","")}</div>'
                    f'    <div style="font-size:13px;color:#111827;margin-top:2px">{fix.get("action","")}</div>'
                    f'  </div>'
                    f'  <div style="display:flex;flex-direction:column;gap:4px;align-items:flex-end">'
                    f'    <span style="font-size:11px;font-weight:600;color:{IMPACT_COLOR[imp]};background:{IMPACT_COLOR[imp]}18;padding:2px 8px;border-radius:999px">⬆ {imp}</span>'
                    f'    <span style="font-size:11px;font-weight:600;color:{EFFORT_COLOR[eff]};background:{EFFORT_COLOR[eff]}18;padding:2px 8px;border-radius:999px">🔧 {eff} effort</span>'
                    f'  </div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # ── Strengths ──
            strengths = report.get("strengths", [])
            if strengths:
                st.markdown("**Strengths**")
                for s in strengths:
                    st.markdown(f'<div style="font-size:13px;color:#166534;padding:3px 0">✓ {s}</div>', unsafe_allow_html=True)

            st.markdown("---")

            # ── Export ──
            st.markdown("**Export**")
            export_data = {
                "analyzed_at": st.session_state.get("analyzed_at"),
                "model_used": selected_model,
                "images": [{"name": i["name"], "dimensions": f"{i['width']}x{i['height']}"} for i in st.session_state["uploaded_images"]],
                "agents": agent_results,
                "orchestrator": report,
            }
            json_str = json.dumps(export_data, indent=2)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇ Download JSON",
                    data=json_str,
                    file_name=f"design_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with dl2:
                # Build markdown report
                md = [f"# Design Analysis Report\n*{st.session_state.get('analyzed_at','')}*\n"]
                md.append(f"## Overall Score: {overall}/100\n> {verdict}\n")
                md.append(f"**Maturity:** {maturity}\n")
                md.append("## Score Breakdown\n")
                for k in AGENT_ORDER:
                    cfg = AGENTS[k]
                    md.append(f"- {cfg['emoji']} **{cfg['name']}**: {breakdown.get(k, 0)}/100")
                md.append("\n## Priority Fixes\n")
                for fix in report.get("priority_fixes", []):
                    md.append(f"{fix['rank']}. **[{fix['area']}]** {fix['action']} *(impact: {fix.get('impact','?')}, effort: {fix.get('effort','?')})*")
                md.append("\n## Strengths\n")
                for s in report.get("strengths", []):
                    md.append(f"- {s}")
                md_str = "\n".join(md)

                st.download_button(
                    "⬇ Download Markdown",
                    data=md_str,
                    file_name=f"design_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            with st.expander("Preview JSON"):
                st.code(json_str[:3000] + ("…" if len(json_str) > 3000 else ""), language="json")
