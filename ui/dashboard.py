from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import uuid4

import streamlit as st

from routezero.pipeline import RouteZeroPipeline


REMOTE_USD_PER_1K_TOKENS = 0.003
LOCAL_USD_PER_1K_TOKENS = 0.0


@st.cache_resource
def get_pipeline() -> RouteZeroPipeline:
    return RouteZeroPipeline()


def run_async(coro):
    return asyncio.run(coro)


def estimate_tokens(*parts: str) -> int:
    text = " ".join(part for part in parts if part)
    return max(1, round(len(text) / 4))


def format_money(value: float) -> str:
    if value < 0.01:
        return f"${value:.5f}"
    return f"${value:.4f}"


def route_label(route: str, cache_hit: bool = False) -> str:
    if cache_hit or route == "cache":
        return "Cache"
    if route == "local":
        return "Local"
    if route == "remote":
        return "Remote"
    return route.replace("_", " ").title()


def route_class(route: str, cache_hit: bool = False) -> str:
    if cache_hit or route == "cache":
        return "cache"
    if route == "remote":
        return "remote"
    if "fallback" in route:
        return "fallback"
    return "local"


def estimate_cost(route: str, tokens: int, cache_hit: bool, local_rate: float, remote_rate: float) -> float:
    if cache_hit or route == "cache":
        return 0.0
    rate = remote_rate if route == "remote" else local_rate
    return (tokens / 1000) * rate


def metric_card(label: str, value: str, caption: str, tone: str = "neutral") -> str:
    return f"""
    <div class="rz-metric rz-tone-{tone}">
        <div class="rz-metric-label">{label}</div>
        <div class="rz-metric-value">{value}</div>
        <div class="rz-metric-caption">{caption}</div>
    </div>
    """


def status_pill(label: str, route: str, cache_hit: bool = False) -> str:
    cls = route_class(route, cache_hit)
    return f'<span class="rz-pill rz-pill-{cls}"><span></span>{label}</span>'


st.set_page_config(
    page_title="RouteZero",
    page_icon="RZ",
    layout="wide",
    initial_sidebar_state="expanded",
)

pipeline = get_pipeline()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600;700;800&display=swap');

    :root {
        --rz-bg: #030303;
        --rz-bg-grid: rgba(255, 255, 255, 0.035);
        --rz-ink: #f7f7f4;
        --rz-text: #f7f7f4;
        --rz-muted: #a4a4a0;
        --rz-dim: #73736d;
        --rz-line: #292925;
        --rz-panel: rgba(18, 18, 16, 0.88);
        --rz-panel-solid: #141412;
        --rz-panel-raised: #1b1b18;
        --rz-input: #0d0d0c;
        --rz-local: #2dd4bf;
        --rz-remote: #fb7185;
        --rz-cache: #a78bfa;
        --rz-fallback: #facc15;
    }

    html, body, [class*="css"] {
        font-family: "Geist Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
        color: var(--rz-text);
        letter-spacing: 0;
    }

    .stApp {
        background:
            linear-gradient(var(--rz-bg-grid) 1px, transparent 1px),
            linear-gradient(90deg, var(--rz-bg-grid) 1px, transparent 1px),
            var(--rz-bg);
        background-size: 28px 28px;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    [data-testid="stSidebar"] {
        background: rgba(9, 9, 8, 0.94);
        border-right: 1px solid var(--rz-line);
        backdrop-filter: blur(18px);
    }

    h1, h2, h3, h4, h5, h6, p, li, label,
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p {
        letter-spacing: 0;
        color: var(--rz-text);
    }

    .stCaptionContainer, .stCaptionContainer p,
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
        color: var(--rz-muted);
    }

    code, pre {
        color: #fefce8 !important;
        background: #10100e !important;
        border: 1px solid var(--rz-line);
    }

    .rz-shell {
        animation: rz-enter 520ms ease both;
    }

    .rz-topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.2rem;
    }

    .rz-brand {
        display: flex;
        align-items: center;
        gap: 0.9rem;
    }

    .rz-logo {
        width: 46px;
        height: 46px;
        display: grid;
        place-items: center;
        border: 1px solid var(--rz-ink);
        color: var(--rz-ink);
        background: var(--rz-panel-solid);
        font-family: "Geist Mono", "SFMono-Regular", Consolas, monospace;
        font-weight: 700;
        box-shadow: 5px 5px 0 #000000;
    }

    .rz-title {
        margin: 0;
        font-size: clamp(2rem, 4vw, 4.25rem);
        line-height: 0.95;
        font-weight: 800;
    }

    .rz-subtitle {
        color: var(--rz-muted);
        margin-top: 0.25rem;
        font-size: 0.9rem;
    }

    .rz-live {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        border: 1px solid var(--rz-line);
        background: var(--rz-panel);
        padding: 0.55rem 0.75rem;
        font-size: 0.78rem;
        color: var(--rz-muted);
    }

    .rz-live span {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: #16a34a;
        animation: rz-pulse 1.6s ease-in-out infinite;
    }

    .rz-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 1rem 0 1.25rem;
    }

    .rz-metric {
        min-height: 128px;
        border: 1px solid var(--rz-line);
        background: var(--rz-panel);
        padding: 1rem;
        position: relative;
        overflow: hidden;
        animation: rz-rise 460ms ease both;
    }

    .rz-metric:after {
        content: "";
        position: absolute;
        inset: auto 0 0;
        height: 3px;
        background: var(--rz-ink);
        transform-origin: left;
        animation: rz-scan 1s ease both;
    }

    .rz-tone-local:after { background: var(--rz-local); }
    .rz-tone-remote:after { background: var(--rz-remote); }
    .rz-tone-cache:after { background: var(--rz-cache); }

    .rz-metric-label {
        text-transform: uppercase;
        color: var(--rz-muted);
        font-size: 0.68rem;
        font-weight: 700;
    }

    .rz-metric-value {
        margin-top: 0.65rem;
        color: var(--rz-ink);
        font-family: "Geist Mono", "SFMono-Regular", Consolas, monospace;
        font-size: clamp(1.45rem, 2.8vw, 2.35rem);
        font-weight: 800;
        line-height: 1;
        white-space: nowrap;
    }

    .rz-metric-caption {
        margin-top: 0.75rem;
        color: var(--rz-muted);
        font-size: 0.76rem;
    }

    .rz-panel {
        border: 1px solid var(--rz-line);
        background: var(--rz-panel);
        padding: 1rem;
        animation: rz-rise 520ms ease both;
    }

    .rz-route-row {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin: 0.35rem 0;
    }

    .rz-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        border: 1px solid currentColor;
        background: #080807;
        padding: 0.28rem 0.55rem;
        text-transform: uppercase;
        font-size: 0.66rem;
        font-weight: 800;
        line-height: 1;
    }

    .rz-pill span {
        width: 6px;
        height: 6px;
        border-radius: 999px;
        background: currentColor;
        animation: rz-pulse 1.6s ease-in-out infinite;
    }

    .rz-pill-local { color: var(--rz-local); }
    .rz-pill-remote { color: var(--rz-remote); }
    .rz-pill-cache { color: var(--rz-cache); }
    .rz-pill-fallback { color: var(--rz-fallback); }

    .rz-bar {
        height: 8px;
        background: #090908;
        border: 1px solid var(--rz-line);
        overflow: hidden;
        margin-top: 0.35rem;
    }

    .rz-bar > div {
        height: 100%;
        background: var(--rz-ink);
        transform-origin: left;
        animation: rz-scan 900ms ease both;
    }

    .rz-bar-local > div { background: var(--rz-local); }
    .rz-bar-remote > div { background: var(--rz-remote); }
    .rz-bar-cache > div { background: var(--rz-cache); }
    .rz-bar-fallback > div { background: var(--rz-fallback); }

    [data-testid="stChatMessage"] {
        background: rgba(20, 20, 18, 0.94);
        border: 1px solid var(--rz-line);
        border-radius: 0;
        animation: rz-rise 320ms ease both;
        color: var(--rz-text);
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: var(--rz-text);
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        color: var(--rz-text);
    }

    [data-testid="stChatMessage"] .rz-pill-local { color: var(--rz-local); }
    [data-testid="stChatMessage"] .rz-pill-remote { color: var(--rz-remote); }
    [data-testid="stChatMessage"] .rz-pill-cache { color: var(--rz-cache); }
    [data-testid="stChatMessage"] .rz-pill-fallback { color: var(--rz-fallback); }
    [data-testid="stChatMessage"] .rz-pill span { color: currentColor; }

    [data-testid="stChatInput"] {
        border-radius: 0;
    }

    [data-testid="stChatInput"] textarea {
        border: 1px solid #4b4b45;
        border-radius: 0;
        background: var(--rz-input);
        color: var(--rz-text);
        box-shadow: 4px 4px 0 #000000;
        caret-color: var(--rz-cache);
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--rz-dim);
        opacity: 1;
    }

    .stButton > button {
        border-radius: 0;
        border: 1px solid #44443f;
        background: var(--rz-panel-raised);
        color: var(--rz-text);
        box-shadow: 3px 3px 0 #000000;
        transition: transform 160ms ease, box-shadow 160ms ease;
    }

    .stButton > button:hover {
        transform: translate(2px, 2px);
        box-shadow: 1px 1px 0 #000000;
        border-color: var(--rz-muted);
        color: var(--rz-text);
        background: #22221f;
    }

    .stNumberInput input {
        background: var(--rz-input);
        color: var(--rz-text);
        border: 1px solid #3a3a35;
        border-radius: 0;
    }

    div[data-baseweb="input"] {
        background: var(--rz-input);
    }

    .stAlert {
        background: #18130d;
        color: #fff7ed;
        border: 1px solid #7c2d12;
    }

    .rz-footnote {
        color: var(--rz-muted);
        font-size: 0.72rem;
        line-height: 1.45;
    }

    @keyframes rz-enter {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes rz-rise {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes rz-pulse {
        0%, 100% { opacity: 0.35; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.18); }
    }

    @keyframes rz-scan {
        from { transform: scaleX(0); }
        to { transform: scaleX(1); }
    }

    @media (max-width: 920px) {
        .rz-topline {
            align-items: flex-start;
            flex-direction: column;
        }
        .rz-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 560px) {
        .rz-grid {
            grid-template-columns: 1fr;
        }
        .rz-metric-value {
            white-space: normal;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid4().hex

if "usage" not in st.session_state:
    st.session_state.usage = {
        "requests": 0,
        "cache_hits": 0,
        "tokens": 0,
        "cost": 0.0,
        "saved_tokens": 0,
        "routes": {},
        "last_route": "idle",
        "last_latency_ms": 0.0,
        "last_model": "",
        "last_updated": None,
    }


if not st.session_state.pipeline_ready:
    with st.empty().container():
        st.markdown(
            """
            <div class="rz-shell" style="min-height:72vh;display:grid;place-items:center;">
                <div class="rz-panel" style="width:min(520px,90vw);text-align:center;padding:2rem;">
                    <div class="rz-logo" style="margin:0 auto 1.3rem;">RZ</div>
                    <h1 class="rz-title" style="font-size:2.4rem;">RouteZero</h1>
                    <p class="rz-subtitle">Initializing semantic cache, router, and model clients.</p>
                    <div class="rz-live" style="margin-top:1.25rem;"><span></span>Warming pipeline</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            run_async(pipeline.init())
            st.session_state.pipeline_ready = True
            st.rerun()
        except Exception as exc:
            st.error(f"RouteZero failed to initialize: {exc}")
            st.stop()


with st.sidebar:
    st.markdown(
        """
        <div class="rz-brand" style="margin:0.5rem 0 1rem;">
            <div class="rz-logo">RZ</div>
            <div>
                <div style="font-weight:800;font-size:1.1rem;">RouteZero</div>
                <div class="rz-footnote">Adaptive LLM routing</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("Cost model")
    remote_rate = st.number_input(
        "Remote USD / 1K tokens",
        min_value=0.0,
        value=REMOTE_USD_PER_1K_TOKENS,
        step=0.001,
        format="%.4f",
    )
    local_rate = st.number_input(
        "Local USD / 1K tokens",
        min_value=0.0,
        value=LOCAL_USD_PER_1K_TOKENS,
        step=0.001,
        format="%.4f",
    )
    st.markdown(
        '<p class="rz-footnote">Token counts use provider usage when available. Cost is estimated from the rates above.</p>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("Route distribution")
    routes = st.session_state.usage["routes"]
    total_routes = max(1, sum(routes.values()))
    for route in ["local", "remote", "cache", "fallback"]:
        count = routes.get(route, 0)
        pct = (count / total_routes) * 100 if routes else 0
        label = route.title()
        st.markdown(
            f"""
            <div style="margin-bottom:0.75rem;">
                <div class="rz-route-row">
                    {status_pill(label, route)}
                    <span style="margin-left:auto;color:var(--rz-muted);font-size:0.75rem;">{count} / {pct:.0f}%</span>
                </div>
                <div class="rz-bar rz-bar-{route}"><div style="width:{pct:.1f}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("New session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = uuid4().hex
            st.rerun()
    with col_b:
        if st.button("Reset all", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = uuid4().hex
            st.session_state.usage = {
                "requests": 0,
                "cache_hits": 0,
                "tokens": 0,
                "cost": 0.0,
                "saved_tokens": 0,
                "routes": {},
                "last_route": "idle",
                "last_latency_ms": 0.0,
                "last_model": "",
                "last_updated": None,
            }
            if pipeline.cache is not None:
                pipeline.cache.clear()
            pipeline.conversation.clear_session(st.session_state.session_id)
            st.rerun()


usage = st.session_state.usage
requests = usage["requests"]
cache_rate = (usage["cache_hits"] / requests * 100) if requests else 0.0
last_updated = usage["last_updated"] or "No requests yet"

st.markdown('<div class="rz-shell">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="rz-topline">
        <div class="rz-brand">
            <div class="rz-logo">RZ</div>
            <div>
                <h1 class="rz-title">RouteZero</h1>
                <div class="rz-subtitle">Minimal adaptive routing console for local, remote, and cached LLM calls.</div>
            </div>
        </div>
        <div class="rz-live"><span></span>Live metrics - {last_updated}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="rz-grid">
        {metric_card("Tokens used", f"{usage['tokens']:,}", "Cumulative session usage", "local")}
        {metric_card("Estimated cost", format_money(usage["cost"]), "Based on sidebar rates", "remote")}
        {metric_card("Cache hit rate", f"{cache_rate:.0f}%", f"{usage['cache_hits']} cache hits", "cache")}
        {metric_card("Requests", f"{requests:,}", f"Last route: {route_label(usage['last_route'])}")}
    </div>
    """,
    unsafe_allow_html=True,
)

detail_cols = st.columns([1.2, 1, 1])
with detail_cols[0]:
    st.markdown(
        f"""
        <div class="rz-panel">
            <div class="rz-metric-label">Current route</div>
            <div style="margin-top:0.8rem;">{status_pill(route_label(usage["last_route"]), usage["last_route"])}</div>
            <div class="rz-footnote" style="margin-top:0.8rem;">Model: {usage["last_model"] or "Waiting for first request"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with detail_cols[1]:
    st.markdown(
        f"""
        <div class="rz-panel">
            <div class="rz-metric-label">Latency</div>
            <div class="rz-metric-value" style="font-size:1.8rem;">{usage["last_latency_ms"]:.0f} ms</div>
            <div class="rz-footnote">Last completed run</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with detail_cols[2]:
    st.markdown(
        f"""
        <div class="rz-panel">
            <div class="rz-metric-label">Saved tokens</div>
            <div class="rz-metric-value" style="font-size:1.8rem;">{usage["saved_tokens"]:,}</div>
            <div class="rz-footnote">Estimated cache savings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

chat_area = st.container()
for message in st.session_state.messages:
    with chat_area.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.markdown(
                f"""
                <div style="margin-top:0.7rem;display:flex;gap:0.5rem;flex-wrap:wrap;">
                    {status_pill(message.get("route_label", "Route"), message.get("route", "local"), message.get("cache_hit", False))}
                    <span class="rz-pill" style="color:var(--rz-muted);"><span></span>{message.get("tokens", 0):,} tokens</span>
                    <span class="rz-pill" style="color:var(--rz-muted);"><span></span>{format_money(message.get("cost", 0.0))}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


if prompt := st.chat_input("Ask RouteZero..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_area.chat_message("user"):
        st.markdown(prompt)

    with chat_area.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(
            """
            <div class="rz-panel" style="display:flex;align-items:center;gap:0.8rem;">
                <div class="rz-live"><span></span>Routing</div>
                <div class="rz-footnote">Checking cache, selecting model, verifying output.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            result = run_async(pipeline.run(prompt, st.session_state.session_id))
        except Exception as exc:
            placeholder.empty()
            st.error(f"Request failed: {exc}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Request failed: {exc}"}
            )
            st.stop()

        placeholder.empty()
        st.markdown(result.response)

        meta = result.metadata
        reported_tokens = int(meta.get("tokens_used") or 0)
        tokens = reported_tokens or estimate_tokens(prompt, result.response)
        cost = estimate_cost(result.route, tokens, result.cache_hit, local_rate, remote_rate)
        label = route_label(result.route, result.cache_hit)
        route_key = route_class(result.route, result.cache_hit)

        st.markdown(
            f"""
            <div style="margin-top:0.7rem;display:flex;gap:0.5rem;flex-wrap:wrap;">
                {status_pill(label, result.route, result.cache_hit)}
                <span class="rz-pill" style="color:var(--rz-muted);"><span></span>{tokens:,} tokens</span>
                <span class="rz-pill" style="color:var(--rz-muted);"><span></span>{format_money(cost)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    usage["requests"] += 1
    usage["tokens"] += tokens
    usage["cost"] += cost
    usage["cache_hits"] += 1 if result.cache_hit else 0
    usage["saved_tokens"] += tokens if result.cache_hit else 0
    usage["routes"][route_key] = usage["routes"].get(route_key, 0) + 1
    usage["last_route"] = route_key
    usage["last_latency_ms"] = float(meta.get("total_pipeline_latency_ms") or meta.get("execution_latency_ms") or 0.0)
    usage["last_model"] = meta.get("model_name") or ("Semantic cache" if result.cache_hit else result.route)
    usage["last_updated"] = datetime.now().strftime("%H:%M:%S")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.response,
            "route": result.route,
            "route_label": label,
            "cache_hit": result.cache_hit,
            "tokens": tokens,
            "cost": cost,
        }
    )

    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
