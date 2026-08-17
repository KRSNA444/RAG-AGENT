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

st.set_page_config(page_title="RAG Agent", page_icon="◈", layout="centered")

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">

<style>
:root {
    --bg: #0b0f1a;
    --bg-glow: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(96, 100, 255, 0.15), transparent);
    --surface: #131829;
    --surface-2: #1a2036;
    --border: #262c45;
    --text: #e6e9f5;
    --text-dim: #8890b5;
    --accent: #7c85ff;
    --accent-glow: rgba(124, 133, 255, 0.35);
    --doc: #34d399;
    --web: #38bdf8;
    --calc: #fbbf24;
    --none: #8890b5;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: var(--bg);
    background-image: var(--bg-glow);
}

section[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}

/* ---- Header ---- */
.agent-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 4px;
    animation: fadeDown 0.5s ease;
}
.agent-header .mark {
    width: 42px; height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--accent), #4f46e5);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; font-weight: 700; color: white;
    box-shadow: 0 0 24px var(--accent-glow);
    flex-shrink: 0;
}
.agent-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, #ffffff, #aab0e8 60%, var(--accent));
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 6s linear infinite;
}
.agent-sub {
    color: var(--text-dim); font-size: 14.5px; margin: 6px 0 26px 0;
    animation: fadeDown 0.6s ease;
}
@keyframes shimmer { to { background-position: 200% center; } }
@keyframes fadeDown { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }

/* ---- Chat bubbles ---- */
[data-testid="stChatMessage"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border);
    border-radius: 14px !important;
    animation: riseIn 0.35s ease;
}
@keyframes riseIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ---- Tool badge with pulsing status dot ---- */
.tool-badge {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 4px 12px 4px 10px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px; font-weight: 600; letter-spacing: 0.02em;
    margin-bottom: 10px; border: 1px solid transparent;
}
.tool-badge .dot {
    width: 7px; height: 7px; border-radius: 50%;
    animation: pulse 1.8s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 currentColor; }
    50% { opacity: 0.6; box-shadow: 0 0 0 4px transparent; }
}
.badge-doc { background: rgba(52,211,153,0.1); color: var(--doc); border-color: rgba(52,211,153,0.25); }
.badge-web { background: rgba(56,189,248,0.1); color: var(--web); border-color: rgba(56,189,248,0.25); }
.badge-calc { background: rgba(251,191,36,0.1); color: var(--calc); border-color: rgba(251,191,36,0.25); }
.badge-none { background: rgba(136,144,181,0.1); color: var(--none); border-color: rgba(136,144,181,0.25); }

/* ---- Suggestion chips (empty state) ---- */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
div[data-testid="stButton"] > button {
    background-color: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-size: 13.5px !important;
    padding: 8px 14px !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 14px var(--accent-glow) !important;
    transform: translateY(-1px);
}

/* ---- Chat input ---- */
[data-testid="stChatInput"] {
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    transition: box-shadow 0.25s ease, border-color 0.25s ease;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* ---- Source expander ---- */
.streamlit-expanderHeader {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12.5px !important;
}

/* ---- Sidebar section labels ---- */
section[data-testid="stSidebar"] h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

DATA_DIR = "data"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("◈ Knowledge Base")

    existing_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if existing_files:
        for f in existing_files:
            st.markdown(f"📄 &nbsp;{f}")
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

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Model · openai/gpt-oss-20b (Groq)")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="agent-header">
    <div class="mark">◈</div>
    <h1>RAG Agent</h1>
</div>
<p class="agent-sub">Ask about your documents, the web, or run a quick calculation — I'll pick the right tool.</p>
""", unsafe_allow_html=True)

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

    SYSTEM_PROMPT = """You are a helpful assistant with access to the user's uploaded documents.
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


def ask_agent(chat_history, retries=2):
    for attempt in range(retries):
        try:
            result = agent.invoke({"messages": chat_history})
            messages = result["messages"]

            tool_used = None
            sources = []
            for msg in messages:
                if hasattr(msg, "name") and msg.name:
                    tool_used = msg.name
                    if msg.name == "search_documents":
                        sources.append(msg.content)

            answer = messages[-1].content
            return answer, tool_used, sources
        except Exception as e:
            error_str = str(e)
            if ("tool_use_failed" in error_str or "rate_limit" in error_str or "429" in error_str) and attempt < retries - 1:
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


# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ---------------------------------------------------------------------------
# Empty state — suggestion chips
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    suggestions = [
        "What is HTML used for?",
        "Explain CSS selectors",
        "45 * 12 kitna hota hai?",
    ]
    cols = st.columns(len(suggestions))
    for col, s in zip(cols, suggestions):
        with col:
            if st.button(s, use_container_width=True, key=f"chip_{s}"):
                st.session_state.pending_prompt = s

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
typed_prompt = st.chat_input("Ask something...")
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
        with st.spinner("Thinking..."):
            answer, tool_used, sources = ask_agent(chat_history)
        st.markdown(tool_badge_html(tool_used), unsafe_allow_html=True)
        st.markdown(answer)
        if tool_used == "search_documents" and sources:
            render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tool_used": tool_used,
        "sources": sources
    })
    st.rerun()