import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from rag_engine import RAGEngine

load_dotenv()

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "TAVILY_API_KEY" in st.secrets:
        os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="RAG Agent", page_icon="◈", layout="wide")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">

<style>
:root {
    --bg: #050407;
    --panel: #0d0a12;
    --card: #14101b;
    --border: rgba(255,255,255,0.08);
    --text: #f2eef7;
    --dim: #8d8698;
    --accent: #a855f7;
    --accent2: #6366f1;
    --glow: rgba(168,85,247,0.35);
    --doc: #34d399; --web: #38bdf8; --calc: #fbbf24; --none: #8d8698;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* hide default chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

/* ---- Animated background ---- */
.stApp {
    background-color: var(--bg);
    background-image:
        radial-gradient(120% 80% at 85% -10%, rgba(168,85,247,0.28), transparent 55%),
        radial-gradient(100% 70% at 0% 100%, rgba(99,102,241,0.16), transparent 60%);
    background-size: 140% 140%, 140% 140%;
    animation: bgDrift 16s ease-in-out infinite alternate;
}
@keyframes bgDrift {
    from { background-position: 0% 0%, 100% 100%; }
    to   { background-position: 30% 20%, 70% 80%; }
}

.block-container { max-width: 780px; padding-top: 2rem; padding-bottom: 7rem; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background-color: #08060c;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] h2 {
    font-family: 'Space Grotesk', sans-serif; font-size: 18px;
}
.doc-item {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px; border-radius: 10px; font-size: 14px; color: #cfc8d9;
    transition: all .25s ease;
}
.doc-item:hover { background: rgba(255,255,255,0.06); transform: translateX(4px); color: #fff; }
.doc-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #22c55e; box-shadow: 0 0 8px #22c55e;
    animation: pulse 2s infinite;
}
.brand-logo {
    width: 34px; height: 34px; border-radius: 50%;
    background: conic-gradient(from 0deg, var(--accent), var(--accent2), #22d3ee, var(--accent));
    animation: spin 8s linear infinite;
    display: inline-block; vertical-align: middle; margin-right: 10px;
    -webkit-mask: radial-gradient(circle, transparent 9px, #000 10px);
            mask: radial-gradient(circle, transparent 9px, #000 10px);
}

/* ---- Hero (empty state) ---- */
.hero { text-align: center; padding: 6vh 0 28px; animation: fadeUp .8s ease both; }
.orb {
    width: 84px; height: 84px; margin: 0 auto 26px; border-radius: 50%; position: relative;
    background: radial-gradient(circle at 30% 28%, #f0abfc, #a855f7 35%, #4338ca 70%, #1e1b4b);
    box-shadow: 0 0 60px rgba(168,85,247,0.6), inset 0 0 20px rgba(255,255,255,0.25);
    animation: float 4s ease-in-out infinite;
}
.orb::after {
    content: ""; position: absolute; inset: -14px; border-radius: 50%;
    border: 1px solid rgba(168,85,247,0.4); animation: ring 3s ease-out infinite;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif; font-size: 34px; font-weight: 600;
    letter-spacing: -0.5px; margin: 0;
    background: linear-gradient(90deg, #fff, #d8b4fe, #fff);
    background-size: 200% auto;
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 5s linear infinite;
}
.hero p { color: var(--dim); margin-top: 10px; font-size: 15px; }

/* ---- Compact header (after first message) ---- */
.mini-header { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; animation: fadeDown .5s ease; }
.mini-orb {
    width: 30px; height: 30px; border-radius: 50%;
    background: radial-gradient(circle at 30% 28%, #f0abfc, #a855f7 40%, #4338ca);
    box-shadow: 0 0 20px rgba(168,85,247,0.6); animation: float 4s ease-in-out infinite;
}
.mini-header span { font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 600; }

/* ---- Feature cards ---- */
.card-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 22px; }
.fcard {
    padding: 16px; border-radius: 16px; background: rgba(13,10,18,0.8);
    border: 1px solid var(--border); transition: all .3s ease; animation: fadeUp .7s both;
}
.fcard:nth-child(1) { animation-delay: .35s; }
.fcard:nth-child(2) { animation-delay: .5s; }
.fcard:nth-child(3) { animation-delay: .65s; }
.fcard:hover { transform: translateY(-6px); border-color: var(--accent); box-shadow: 0 12px 30px rgba(168,85,247,0.25); }
.fcard .tag {
    float: right; font-size: 11px; padding: 4px 10px; border-radius: 999px;
    background: rgba(255,255,255,0.08); font-family: 'JetBrains Mono', monospace;
}
.fcard h5 { margin: 28px 0 0; font-size: 14.5px; font-weight: 500; color: var(--text); }
.fcard p { margin: 6px 0 0; font-size: 13px; color: var(--dim); line-height: 1.5; }

/* ---- Chat messages (ChatGPT / Claude style) ---- */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    backdrop-filter: none !important;
    padding: 6px 0 !important;
    animation: msgIn .4s cubic-bezier(.2,.8,.2,1);
}

/* hide user avatar only */
[data-testid="stChatMessageAvatarUser"] { display: none !important; }

/* user message: right-aligned bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
    margin-left: auto;
    width: fit-content;
    max-width: 75%;
    background: rgba(255,255,255,0.08) !important;
    border-radius: 20px 20px 4px 20px !important;
    padding: 10px 18px !important;
}

/* assistant message: plain text, a bit larger and airy */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    font-size: 16px;
    line-height: 1.75;
}

/* ---- Gemini-style animated assistant avatar ---- */
[data-testid="stChatMessageAvatarAssistant"] {
    display: flex !important;
    position: relative;
    width: 34px; height: 34px;
    flex-shrink: 0;
    background: transparent !important;
    color: transparent !important;
    font-size: 0 !important;
}
/* hide the default emoji / icon */
[data-testid="stChatMessageAvatarAssistant"] > * { display: none !important; }

/* rotating colour glow behind the star */
[data-testid="stChatMessageAvatarAssistant"]::before {
    content: "";
    position: absolute; inset: -6px;
    border-radius: 50%;
    background: conic-gradient(from 0deg, #4285f4, #a855f7, #ec4899, #22d3ee, #4285f4);
    filter: blur(10px);
    opacity: .65;
    animation: spin 4s linear infinite;
}
/* 4-point sparkle */
[data-testid="stChatMessageAvatarAssistant"]::after {
    content: "";
    position: absolute; inset: 3px;
    background: linear-gradient(135deg, #8ab4ff, #c084fc, #f472b6);
    clip-path: polygon(50% 0%, 61% 39%, 100% 50%, 61% 61%, 50% 100%, 39% 61%, 0% 50%, 39% 39%);
    animation: sparkle 2.4s ease-in-out infinite;
}

/* ---- Tool badge ---- */
.tool-badge {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 4px 12px 4px 10px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace; font-size: 11.5px; font-weight: 600;
    margin-bottom: 10px; border: 1px solid transparent;
}
.tool-badge .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; animation: pulse 1.8s ease-in-out infinite; }
.badge-doc  { background: rgba(52,211,153,0.1);  color: var(--doc);  border-color: rgba(52,211,153,0.25); }
.badge-web  { background: rgba(56,189,248,0.1);  color: var(--web);  border-color: rgba(56,189,248,0.25); }
.badge-calc { background: rgba(251,191,36,0.1);  color: var(--calc); border-color: rgba(251,191,36,0.25); }
.badge-none { background: rgba(141,134,152,0.1); color: var(--none); border-color: rgba(141,134,152,0.25); }

/* ---- Buttons / chips ---- */
div[data-testid="stButton"] > button {
    background: rgba(0,0,0,0.5) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    color: var(--text) !important;
    font-size: 13.5px !important;
    padding: 8px 16px !important;
    transition: all .25s ease !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 6px 20px var(--glow) !important;
    transform: translateY(-3px);
}

/* ---- Chat input (pill) ---- */
[data-testid="stBottom"] > div { background: transparent !important; }
[data-testid="stChatInput"] {
    border-radius: 999px !important;
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    transition: box-shadow .3s ease, border-color .3s ease;
}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea { background: transparent !important; }
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(168,85,247,0.6) !important;
    box-shadow: 0 0 0 1px rgba(168,85,247,0.4), 0 0 30px rgba(168,85,247,0.2) !important;
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border-radius: 50% !important;
}

/* ---- Status box (agent steps) ---- */
[data-testid="stStatus"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
}
/* shimmering "Agent is thinking…" text, like Gemini's loading state */
[data-testid="stStatus"] summary p {
    background: linear-gradient(90deg, #8d8698 20%, #fff 50%, #8d8698 80%);
    background-size: 200% auto;
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 2.5s linear infinite;
}

/* ---- Expander ---- */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    font-family: 'JetBrains Mono', monospace !important; font-size: 12.5px !important;
}

/* ---- Keyframes ---- */
@keyframes spin   { to { transform: rotate(360deg); } }
@keyframes float  { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-12px); } }
@keyframes ring   { 0% { transform: scale(.9); opacity: .8; } 100% { transform: scale(1.8); opacity: 0; } }
@keyframes shine  { to { background-position: 200% center; } }
@keyframes pulse  { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
@keyframes sparkle { 0%,100% { transform: scale(.9) rotate(0deg); } 50% { transform: scale(1.2) rotate(90deg); } }
@keyframes fadeUp   { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: none; } }
@keyframes fadeDown { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: none; } }
@keyframes msgIn    { from { opacity: 0; transform: translateY(14px) scale(.98); } to { opacity: 1; transform: none; } }

@media (max-width: 700px) { .card-grid { grid-template-columns: 1fr; } .hero h1 { font-size: 26px; } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<h2><span class="brand-logo"></span>RAG Agent</h2>', unsafe_allow_html=True)

    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("##### Knowledge Base")
    existing_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if existing_files:
        for f in existing_files:
            st.markdown(f'<div class="doc-item"><span class="doc-dot"></span>{f}</div>', unsafe_allow_html=True)
    else:
        st.caption("No documents loaded yet.")

    st.divider()

    uploaded_file = st.file_uploader("Add a new PDF", type="pdf")
    if uploaded_file is not None:
        save_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"{uploaded_file.name} added. Rebuild the index below.")

    if st.button("⟳ Rebuild Index", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Model · openai/gpt-oss-20b (Groq)")

# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------
@st.cache_resource
def setup_agent():
    rag = RAGEngine()
    rag.build_index(DATA_DIR)

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_retries=2)

    @tool
    def search_documents(query: str) -> str:
        """ALWAYS use this tool FIRST for any question about topics that might be in the user's documents.
        This searches the user's uploaded PDF documents which contain the authoritative source material.
        Only use web_search if this tool returns no relevant information."""
        chunks = rag.retrieve(query, top_k=2)
        if not chunks:
            return "No relevant information found in documents."
        return "\n\n".join(chunks)

    web_search = TavilySearchResults(max_results=3)
    web_search.name = "web_search"
    web_search.description = "Search the internet for current, real-time, or general knowledge information not found in the user's documents."

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression and return the numeric result.
        Input should be a valid Python math expression, e.g. '2 + 2' or '15 * 3.5'."""
        try:
            result = eval(expression, {"__builtins__": {}})
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error evaluating expression: {e}"

    SYSTEM_PROMPT = """You are RAG Agent, an AI assistant built by Krishnakanth.
    If asked your name or who you are, say you are "RAG Agent", an assistant that can search the user's uploaded documents, the web, and do calculations.
    Never say you are ChatGPT or mention OpenAI.
    ALWAYS respond in English only, regardless of the language the user writes in.
    When answering questions related to topics that might be in the user's documents,
    ALWAYS call search_documents FIRST and base your answer STRICTLY on the retrieved content.
    Do NOT use your own general knowledge to answer if search_documents returns relevant content.
    If search_documents returns no relevant info, then you may use web_search or your own knowledge, but mention that it's not from the user's documents.
    When using the calculator tool, always state the exact numeric result clearly in your final answer."""

    tools = [search_documents, web_search, calculator]
    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


with st.spinner("Booting up the knowledge base..."):
    agent = setup_agent()

TOOL_LABELS = {
    "search_documents": "Searching your documents (FAISS)",
    "web_search": "Searching the web (Tavily)",
    "calculator": "Running calculator",
}


def ask_agent(chat_history, status=None, retries=2):
    """Runs the agent while streaming each tool call into the status box."""
    for attempt in range(retries):
        try:
            messages = None
            for chunk in agent.stream({"messages": chat_history}, stream_mode="values"):
                messages = chunk["messages"]
                calls = getattr(messages[-1], "tool_calls", None)
                if calls and status is not None:
                    for c in calls:
                        status.write(f"🔧 {TOOL_LABELS.get(c['name'], c['name'])}…")

            tool_used = None
            sources = []
            for msg in messages:
                if hasattr(msg, "name") and msg.name:
                    tool_used = msg.name
                    if msg.name == "search_documents":
                        sources.append(msg.content)

            return messages[-1].content, tool_used, sources
        except Exception as e:
            error_str = str(e)
            if ("tool_use_failed" in error_str or "rate_limit" in error_str or "429" in error_str) and attempt < retries - 1:
                if status is not None:
                    status.write("⏳ Rate limited, retrying…")
                time.sleep(8)
                continue
            return f"⚠️ Error: {e}", None, []


def tool_badge_html(tool_used):
    badges = {
        "search_documents": ("Documents", "badge-doc"),
        "web_search": ("Web Search", "badge-web"),
        "calculator": ("Calculator", "badge-calc"),
    }
    label, cls = badges.get(tool_used, ("General Knowledge", "badge-none"))
    return f'<span class="tool-badge {cls}"><span class="dot"></span>{label}</span>'


def render_sources(sources):
    with st.expander("View source chunks"):
        for i, src in enumerate(sources, 1):
            st.markdown(f"**Chunk {i}**")
            st.text(src[:500] + ("..." if len(src) > 500 else ""))


def stream_words(text):
    """Typing effect for the final answer."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ---------------------------------------------------------------------------
# Empty state (hero) or compact header
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown("""
    <div class="hero">
        <div class="orb"></div>
        <h1>Ready to Ask Your Documents?</h1>
        <p>Search your knowledge base, the web, or run a calculation — I'll pick the right tool.</p>
    </div>
    """, unsafe_allow_html=True)

    suggestions = ["What is RAG?", "Explain AI agents", "How does vector search work?"]
    cols = st.columns(len(suggestions))
    for col, s in zip(cols, suggestions):
        with col:
            if st.button(s, use_container_width=True, key=f"chip_{s}"):
                st.session_state.pending_prompt = s

    st.markdown("""
    <div class="card-grid">
        <div class="fcard"><span class="tag">FAISS</span><h5>Document Q&amp;A</h5><p>Answers grounded in your uploaded PDFs, with source chunks.</p></div>
        <div class="fcard"><span class="tag">Tavily</span><h5>Web Search</h5><p>Live results when the answer isn't in your documents.</p></div>
        <div class="fcard"><span class="tag">LangGraph</span><h5>Smart Agent</h5><p>Automatically picks the best tool for every question.</p></div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="mini-header"><div class="mini-orb"></div><span>RAG Agent</span></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Render history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("tool_used") is not None:
            st.markdown(tool_badge_html(msg["tool_used"]), unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg.get("tool_used") == "search_documents" and msg.get("sources"):
            render_sources(msg["sources"])

# ---------------------------------------------------------------------------
# Input handling (chat box OR clicked suggestion chip)
# ---------------------------------------------------------------------------
typed_prompt = st.chat_input("Ask anything...")
prompt = typed_prompt or st.session_state.pending_prompt
st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chat_history = []
    for msg in st.session_state.messages[-10:]:
        role = "user" if msg["role"] == "user" else "assistant"
        chat_history.append((role, msg["content"]))

    with st.chat_message("assistant"):
        with st.status("Agent is thinking…", expanded=True) as status:
            answer, tool_used, sources = ask_agent(chat_history, status)
            status.update(label="Done", state="complete", expanded=False)

        st.markdown(tool_badge_html(tool_used), unsafe_allow_html=True)
        st.write_stream(stream_words(answer))
        if tool_used == "search_documents" and sources:
            render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tool_used": tool_used,
        "sources": sources,
    })
    st.rerun()