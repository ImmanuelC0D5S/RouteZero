import asyncio
import streamlit as st
from routezero.pipeline import RouteZeroPipeline


@st.cache_resource
def get_pipeline():
    """Initialize the pipeline (lazy — heavy model loading deferred)."""
    return RouteZeroPipeline()


pipeline = get_pipeline()

st.set_page_config(
    page_title="RouteZero — Adaptive LLM Router",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Lazy init: show spinner while model downloads on first request ─────
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False

if not st.session_state.pipeline_ready:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:80vh;gap:1rem">
            <div style="font-size:3rem">⚡</div>
            <div style="font-size:1.2rem;font-weight:600;color:var(--text-primary)">RouteZero is warming up...</div>
            <div style="font-size:0.85rem;color:var(--text-muted);text-align:center;max-width:400px">
                Loading the semantic cache model (all-MiniLM-L6-v2).<br>
                This may take 30–60 seconds on first run.
            </div>
            <div style="display:flex;gap:0.5rem;margin-top:1rem">
                <span style="width:12px;height:12px;border-radius:50%;background:var(--neon-cyan);animation:pulseDot 1s ease-in-out infinite"></span>
                <span style="width:12px;height:12px;border-radius:50%;background:var(--neon-purple);animation:pulseDot 1s ease-in-out infinite 0.2s"></span>
                <span style="width:12px;height:12px;border-radius:50%;background:var(--neon-magenta);animation:pulseDot 1s ease-in-out infinite 0.4s"></span>
            </div>
        </div>
        <style>
        @keyframes pulseDot {
            0%, 100% { opacity: 0.3; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.3); }
        }
        </style>
        """, unsafe_allow_html=True)
    try:
        asyncio.run(pipeline.init())
        st.session_state.pipeline_ready = True
        placeholder.empty()
        st.rerun()
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        st.stop()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-deep: #0a0a1a;
    --bg-surface: #0f0f24;
    --bg-card: #14143a;
    --neon-cyan: #00f0ff;
    --neon-magenta: #ff00e5;
    --neon-purple: #8b5cf6;
    --neon-green: #10b981;
    --neon-amber: #f59e0b;
    --neon-red: #ef4444;
    --glow-cyan: 0 0 20px rgba(0, 240, 255, 0.3), 0 0 60px rgba(0, 240, 255, 0.1);
    --glow-magenta: 0 0 20px rgba(255, 0, 229, 0.3), 0 0 60px rgba(255, 0, 229, 0.1);
    --glow-purple: 0 0 20px rgba(139, 92, 246, 0.3), 0 0 60px rgba(139, 92, 246, 0.1);
    --text-primary: #e4e4f0;
    --text-secondary: #9494b8;
    --text-muted: #5a5a7a;
}

* { font-family: 'Inter', sans-serif; }
code, pre, .mono { font-family: 'JetBrains Mono', monospace; }

.stApp {
    background: var(--bg-deep);
    background-image:
        radial-gradient(ellipse 80% 60% at 50% -20%, rgba(0, 240, 255, 0.05) 0%, transparent 70%),
        radial-gradient(ellipse 60% 80% at 80% 80%, rgba(139, 92, 246, 0.04) 0%, transparent 70%),
        radial-gradient(ellipse 60% 80% at 20% 90%, rgba(255, 0, 229, 0.03) 0%, transparent 70%);
}

.main .block-container { max-width: 1000px; padding-top: 1.5rem; }

h1, h2, h3 { font-weight: 800; letter-spacing: -0.02em; color: var(--text-primary); }
h1 { font-size: 2.2rem; }
p, li, .stMarkdown { color: var(--text-secondary); }

.stChatFloatingInputContainer {
    background: var(--bg-surface) !important;
    border-top: 1px solid rgba(0, 240, 255, 0.1) !important;
    backdrop-filter: blur(20px);
}
[data-testid="stChatInput"] input {
    background: #1a1a3a !important;
    border: 1px solid #2a2a5a !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    transition: all 0.3s ease;
}
[data-testid="stChatInput"] input:focus {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.15) !important;
}

.stChatMessage {
    border-radius: 16px !important;
    margin-bottom: 0.5rem;
    animation: msgSlide 0.4s ease-out;
}
@keyframes msgSlide {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
[data-testid="stChatMessage"] {
    border: 1px solid rgba(255,255,255,0.06);
    background: rgba(20, 20, 58, 0.6) !important;
    backdrop-filter: blur(12px);
}
[data-testid="stChatMessage"][data-testid$="user"] {
    background: rgba(0, 240, 255, 0.04) !important;
    border-color: rgba(0, 240, 255, 0.1);
}

.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    background: linear-gradient(135deg, #1a1a4a, #0f0f30) !important;
    border: 1px solid #2a2a5a !important;
    color: var(--text-primary) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0, 240, 255, 0.15) !important;
    border-color: var(--neon-cyan) !important;
}

[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid rgba(0, 240, 255, 0.08);
}
[data-testid="stSidebar"] .stMarkdown { color: var(--text-secondary); }
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.06);
    margin: 1rem 0;
}

.neon-glow-cyan { box-shadow: var(--glow-cyan); }
.neon-glow-magenta { box-shadow: var(--glow-magenta); }
.neon-glow-purple { box-shadow: var(--glow-purple); }

.neon-card {
    background: linear-gradient(135deg, #14143a, #1a1a4a);
    border: 1px solid rgba(0, 240, 255, 0.1);
    border-radius: 16px;
    padding: 1.25rem;
    position: relative;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}
.neon-card:hover {
    border-color: rgba(0, 240, 255, 0.3);
    box-shadow: 0 8px 40px rgba(0, 240, 255, 0.1);
    transform: translateY(-2px);
}
.neon-card::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,240,255,0.03), transparent);
    transition: left 0.6s;
}
.neon-card:hover::before { left: 100%; }

.metric-ring {
    position: relative;
    width: 72px; height: 72px;
    margin: 0 auto;
}
.metric-ring svg { transform: rotate(-90deg); }
.metric-ring .bg { fill: none; stroke: rgba(255,255,255,0.06); stroke-width: 4; }
.metric-ring .progress {
    fill: none; stroke-width: 4; stroke-linecap: round;
    stroke-dasharray: 188.5; stroke-dashoffset: 188.5;
    transition: stroke-dashoffset 1.5s cubic-bezier(0.4, 0, 0.2, 1);
}
.metric-ring .progress.cyan { stroke: var(--neon-cyan); filter: drop-shadow(0 0 6px rgba(0,240,255,0.5)); }
.metric-ring .progress.magenta { stroke: var(--neon-magenta); filter: drop-shadow(0 0 6px rgba(255,0,229,0.5)); }
.metric-ring .progress.purple { stroke: var(--neon-purple); filter: drop-shadow(0 0 6px rgba(139,92,246,0.5)); }
.metric-ring .progress.green { stroke: var(--neon-green); filter: drop-shadow(0 0 6px rgba(16,185,129,0.5)); }
.metric-ring .value {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 1.1rem; font-weight: 700;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
}

.route-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.2rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.65rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace;
    transition: all 0.3s ease;
}
.route-badge:hover { transform: scale(1.05); }
.route-local { background: rgba(0, 240, 255, 0.1); color: var(--neon-cyan); border: 1px solid rgba(0, 240, 255, 0.2); }
.route-remote { background: rgba(255, 0, 229, 0.1); color: var(--neon-magenta); border: 1px solid rgba(255, 0, 229, 0.2); }
.route-fallback_remote, .route-fallback { background: rgba(245, 158, 11, 0.1); color: var(--neon-amber); border: 1px solid rgba(245, 158, 11, 0.2); }
.route-cache { background: rgba(139, 92, 246, 0.1); color: var(--neon-purple); border: 1px solid rgba(139, 92, 246, 0.2); }

.pulse-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    animation: pulseDot 1.5s ease-in-out infinite;
}
@keyframes pulseDot {
    0%, 100% { opacity: 0.4; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.3); }
}

.progress-bar-bg {
    width: 100%; height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 9999px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 9999px;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple));
    box-shadow: 0 0 10px rgba(0,240,255,0.3);
}

.stSpinner > div { border-color: var(--neon-cyan) transparent transparent transparent !important; }

.main-title {
    display: flex; align-items: center; gap: 0.75rem;
    margin-bottom: 0;
}
.main-title h1 {
    background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple), var(--neon-magenta));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5rem;
    line-height: 1;
}
.subtitle {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-top: -0.25rem;
    margin-bottom: 1.5rem;
}

.pipeline-node {
    display: flex; flex-direction: column; align-items: center; gap: 0.4rem;
    position: relative;
}
.pipeline-node .node-circle {
    width: 40px; height: 40px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    border: 2px solid;
    background: rgba(20,20,58,0.8);
    backdrop-filter: blur(8px);
    transition: all 0.3s ease;
}
.pipeline-node .node-circle.active { box-shadow: 0 0 20px rgba(0,240,255,0.3); }
.pipeline-node .node-label {
    font-size: 0.6rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--text-muted);
    text-align: center; line-height: 1.2;
}
.pipeline-edge {
    width: 40px; height: 2px;
    background: linear-gradient(90deg, rgba(0,240,255,0.2), rgba(0,240,255,0.5));
    position: relative;
    align-self: center;
}
.pipeline-edge .flow-particle {
    position: absolute; width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--neon-cyan);
    box-shadow: 0 0 10px rgba(0,240,255,0.6);
    animation: flowRight 1.2s ease-in-out infinite;
}
@keyframes flowRight {
    0% { left: 0; opacity: 0; }
    20% { opacity: 1; }
    80% { opacity: 1; }
    100% { left: calc(100% - 6px); opacity: 0; }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.shimmer-text {
    background: linear-gradient(90deg, var(--text-primary) 0%, var(--neon-cyan) 50%, var(--text-primary) 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s linear infinite;
}

div[data-testid="stVerticalBlock"]:has(> .element-container) { gap: 0.5rem; }

.st-cb, .st-c0, .st-c1, .st-c2, .st-c3, .st-c4, .st-c5, .st-c6, .st-c7, .st-c8, .st-c9 {
    background: transparent !important;
}

.stText { color: var(--text-secondary) !important; }
</style>
""", unsafe_allow_html=True)

PIPELINE_NODES = [
    {"id": "input", "label": "Input", "icon": "⌨️", "color": "#8b5cf6"},
    {"id": "cache", "label": "Semantic<br>Cache", "icon": "💾", "color": "#8b5cf6"},
    {"id": "router", "label": "Router", "icon": "🎯", "color": "#00f0ff"},
    {"id": "model", "label": "Model", "icon": "🧠", "color": "#ff00e5"},
    {"id": "output", "label": "Output", "icon": "✅", "color": "#10b981"},
]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_session"
if "metrics" not in st.session_state:
    st.session_state.metrics = {"cache_hits": 0, "routes": {}, "total_cost": 0.0, "tokens": 0, "history_tokens": 0}
if "last_route" not in st.session_state:
    st.session_state.last_route = None

def route_color(name: str) -> str:
    return {"local": "cyan", "remote": "magenta", "fallback_remote": "amber", "fallback": "amber", "cache": "purple"}.get(name, "cyan")

def route_badge(route: str) -> str:
    cls = route.replace("_", "_")
    label = route.replace("_remote", " ← remote").replace("_", " ")
    return f'<span class="route-badge route-{cls}"><span class="pulse-dot" style="background: var(--neon-{"cyan" if route=="local" else "magenta" if route=="remote" else "amber" if "fallback" in route else "purple"})"></span>{label}</span>'

def pipeline_viz(active: str | None = None) -> str:
    parts = []
    for i, node in enumerate(PIPELINE_NODES):
        is_active = node["id"] == active
        active_cls = " active" if is_active else ""
        parts.append(f'''
        <div class="pipeline-node">
            <div class="node-circle{active_cls}" style="border-color: {node['color']}; box-shadow: {'0 0 20px ' + node['color'] + '40' if is_active else 'none'}">
                {node['icon']}
            </div>
            <div class="node-label">{node['label']}</div>
        </div>''')
        if i < len(PIPELINE_NODES) - 1:
            parts.append(f'''
        <div class="pipeline-edge">
            <div class="flow-particle" style="animation-delay: {i * 0.15}s"></div>
        </div>''')
    return '<div style="display:flex;align-items:center;justify-content:center;gap:0;padding:0.75rem 0">' + ''.join(parts) + '</div>'

def metric_ring(value_pct: float, label: str, color: str, display_val: str) -> str:
    dashoffset = 188.5 - (188.5 * min(value_pct, 1))
    return f'''
    <div class="neon-card" style="text-align:center;padding:1rem">
        <div class="metric-ring">
            <svg width="72" height="72" viewBox="0 0 72 72">
                <circle class="bg" cx="36" cy="36" r="30"/>
                <circle class="progress {color}" cx="36" cy="36" r="30" style="stroke-dashoffset: {dashoffset}"/>
            </svg>
            <div class="value">{display_val}</div>
        </div>
        <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-muted);margin-top:0.5rem;font-weight:600">{label}</div>
    </div>'''

col1, col2, col3 = st.columns([1, 0.05, 2.5])

with col1:
    st.markdown(f'<div class="main-title"><h1>RouteZero</h1></div>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">⚡ Adaptive LLM Routing Engine</p>', unsafe_allow_html=True)

with col3:
    total_reqs = max(sum(st.session_state.metrics["routes"].values()), 1)
    cache_rate = st.session_state.metrics["cache_hits"] / total_reqs
    st.markdown(metric_ring(cache_rate, "Cache Rate", "cyan", f"{cache_rate*100:.0f}%"), unsafe_allow_html=True)

mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.markdown(metric_ring(
        min(st.session_state.metrics["cache_hits"] / max(total_reqs, 1) * 2, 1),
        "Cache Hits", "purple", str(st.session_state.metrics["cache_hits"])
    ), unsafe_allow_html=True)
with mcol2:
    max_cost = max(st.session_state.metrics["total_cost"], 0.001)
    st.markdown(metric_ring(
        min(st.session_state.metrics["total_cost"] / 0.1, 1),
        "Total Cost", "magenta", f"${st.session_state.metrics['total_cost']:.4f}"
    ), unsafe_allow_html=True)
with mcol3:
    max_tokens = max(st.session_state.metrics["tokens"], 1000)
    st.markdown(metric_ring(
        min(st.session_state.metrics["tokens"] / max_tokens, 1),
        "Tokens Used", "cyan", f"{st.session_state.metrics['tokens']:,}"
    ), unsafe_allow_html=True)
with mcol4:
    st.markdown(metric_ring(
        1.0, "Requests", "green", str(total_reqs - 1)
    ), unsafe_allow_html=True)

st.markdown("---")

pviz = pipeline_viz(active=st.session_state.last_route)
st.markdown(f'<div style="background:linear-gradient(135deg,#0f0f24,#14143a);border-radius:16px;border:1px solid rgba(0,240,255,0.08);padding:0.5rem 1rem;margin-bottom:1rem">{pviz}</div>', unsafe_allow_html=True)

chat_container = st.container()

for msg in st.session_state.messages:
    with chat_container.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "route" in msg:
            st.markdown(f'<div style="margin-top:0.5rem">{route_badge(msg["route"])}</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f'''
    <div style="text-align:center;padding:1rem 0">
        <div style="font-size:2.5rem;margin-bottom:0.25rem">⚡</div>
        <div style="font-size:1.1rem;font-weight:800;background:linear-gradient(135deg,var(--neon-cyan),var(--neon-purple),var(--neon-magenta));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">RouteZero</div>
        <div style="font-size:0.7rem;color:var(--text-muted);font-weight:500;letter-spacing:0.05em;text-transform:uppercase">Adaptive LLM Router</div>
    </div>''', unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown(f'''
    <div class="neon-card" style="padding:0.85rem">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
            <span style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:var(--text-muted)">Pipeline Status</span>
            <span class="route-badge route-local" style="font-size:0.55rem"><span class="pulse-dot" style="background:var(--neon-cyan)"></span>Online</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem">
            <div style="display:flex;justify-content:space-between;font-size:0.7rem"><span style="color:var(--text-muted)">Cache</span><span style="color:var(--neon-purple);font-weight:600;font-family:'JetBrains Mono',monospace">{'🟢 Ready' if st.session_state.metrics['cache_hits'] > 0 else '⚪ Idle'}</span></div>
            <div style="display:flex;justify-content:space-between;font-size:0.7rem"><span style="color:var(--text-muted)">Router</span><span style="color:var(--neon-cyan);font-weight:600;font-family:'JetBrains Mono',monospace">🟢 Active</span></div>
            <div style="display:flex;justify-content:space-between;font-size:0.7rem"><span style="color:var(--text-muted)">Local (Groq)</span><span style="color:var(--neon-cyan);font-weight:600;font-family:'JetBrains Mono',monospace">🟢 Ready</span></div>
            <div style="display:flex;justify-content:space-between;font-size:0.7rem"><span style="color:var(--text-muted)">Remote (Gemini)</span><span style="color:var(--neon-magenta);font-weight:600;font-family:'JetBrains Mono',monospace">🟢 Connected</span></div>
        </div>
    </div>''', unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-muted);margin-bottom:0.5rem">Route Distribution</div>', unsafe_allow_html=True)

    all_routes = ["local", "remote", "fallback_remote", "cache"]
    route_labels = {"local": "Groq (local)", "remote": "Gemini (remote)", "fallback_remote": "Fallback", "cache": "Cache"}
    for route in all_routes:
        count = st.session_state.metrics["routes"].get(route, 0)
        pct = (count / (total_reqs - 1) * 100) if total_reqs > 1 else 0
        bar_color = "cyan" if route == "local" else "magenta" if route == "remote" else "amber" if "fallback" in route else "purple"
        st.markdown(f'''
        <div style="margin-bottom:0.6rem">
            <div style="display:flex;justify-content:space-between;font-size:0.7rem;margin-bottom:0.25rem">
                <span>{route_badge(route)}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--text-primary)">{count} <span style="color:var(--text-muted);font-weight:400">({pct:.0f}%)</span></span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,var(--neon-{bar_color}),var(--neon-{bar_color}))"></div>
            </div>
        </div>''', unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)

    if st.button("⟳ Clear & Reset", use_container_width=True):
        st.session_state.messages = []
        st.session_state.metrics = {"cache_hits": 0, "routes": {}, "total_cost": 0.0, "tokens": 0, "history_tokens": 0}
        st.session_state.last_route = None
        # Wipe ChromaDB cache so old responses aren't returned
        pipeline.cache.clear()
        # Wipe conversation history so new queries aren't polluted by old context
        pipeline.conversation.clear_session(st.session_state.session_id)
        st.rerun()

    st.markdown(f'''
    <div style="margin-top:1rem;padding:0.75rem;border-radius:10px;background:rgba(0,240,255,0.03);border:1px solid rgba(0,240,255,0.06);text-align:center">
        <div style="font-size:0.5rem;color:var(--text-muted);letter-spacing:0.03em;line-height:1.4">
            <div>🖥 Groq · llama-3.3-70b</div>
            <div>☁ Google Gemini 2.0 Flash</div>
            <div style="margin-top:0.3rem">⚡ v1.0 · RouteZero</div>
        </div>
    </div>''', unsafe_allow_html=True)

if prompt := st.chat_input("Ask anything...", key="chat_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with chat_container.chat_message("user"):
        st.markdown(prompt)

    with chat_container.chat_message("assistant"):
        with st.spinner(""):
            placeholder = st.empty()
            placeholder.markdown(f'''
            <div style="display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0">
                <div style="font-size:1.2rem">⚡</div>
                <div>
                    <div style="font-size:0.8rem;font-weight:600;color:var(--text-muted)">Routing through pipeline...</div>
                    <div style="display:flex;gap:0.4rem;margin-top:0.3rem;align-items:center">
                        <span style="width:8px;height:8px;border-radius:50%;background:var(--neon-cyan);animation:pulseDot 1s ease-in-out infinite"></span>
                        <span style="width:8px;height:8px;border-radius:50%;background:var(--neon-purple);animation:pulseDot 1s ease-in-out infinite 0.2s"></span>
                        <span style="width:8px;height:8px;border-radius:50%;background:var(--neon-magenta);animation:pulseDot 1s ease-in-out infinite 0.4s"></span>
                    </div>
                </div>
            </div>''', unsafe_allow_html=True)

            result = asyncio.run(pipeline.run(prompt, st.session_state.session_id))
            placeholder.empty()
            st.markdown(result.response)

        route = result.route
        st.session_state.last_route = route
        st.markdown(f'<div style="margin-top:0.5rem">{route_badge(route)} <span style="font-size:0.6rem;color:var(--text-muted);margin-left:0.5rem">{"💾 Cache hit" if result.cache_hit else "🔄 Fresh route"}</span></div>', unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant", "content": result.response, "route": route
    })

    st.session_state.metrics["routes"][route] = st.session_state.metrics["routes"].get(route, 0) + 1
    if result.cache_hit:
        st.session_state.metrics["cache_hits"] += 1

    meta = result.metadata
    st.session_state.metrics["tokens"] += meta.get("history_tokens", 0)

    st.rerun()
