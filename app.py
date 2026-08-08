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


# Load keys from Streamlit secrets if available (for cloud deployment)
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "TAVILY_API_KEY" in st.secrets:
        os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="RAG Agent", page_icon="🤖", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .stChatMessage { background-color: #1a1c24; border-radius: 10px; }
    .tool-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 12px; margin-bottom: 8px; font-weight: 600;
    }
    .badge-doc { background-color: #1f3a2e; color: #4ade80; }
    .badge-web { background-color: #1e2f4a; color: #60a5fa; }
    .badge-calc { background-color: #3a2f1f; color: #fbbf24; }
    .badge-none { background-color: #2a2a2a; color: #9ca3af; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = "data"

# --- Sidebar ---
with st.sidebar:
    st.header("📁 Documents")

    existing_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if existing_files:
        for f in existing_files:
            st.markdown(f"📄 {f}")
    else:
        st.caption("Koi PDF nahi hai abhi.")

    st.divider()

    uploaded_file = st.file_uploader("Upload a new PDF", type="pdf")
    if uploaded_file is not None:
        save_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"{uploaded_file.name} added! Rebuild index karo neeche button se.")

    if st.button("🔄 Rebuild Index", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.messages = []
        st.rerun()

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Model: openai/gpt-oss-20b (Groq)")

# --- Main title ---
st.title("🤖 RAG Agent")
st.caption("Ask questions about your documents, or anything else — I'll figure out the right tool to use.")

# --- Cache the agent setup ---
@st.cache_resource
def setup_agent():
    rag = RAGEngine()
    rag.build_index(DATA_DIR)

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_retries=2
    )

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

agent = setup_agent()

def ask_agent(query, retries=2):
    for attempt in range(retries):
        try:
            result = agent.invoke({"messages": [("user", query)]})
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
        "search_documents": ("📄 From Documents", "badge-doc"),
        "web_search": ("🌐 From Web Search", "badge-web"),
        "calculator": ("🔢 From Calculator", "badge-calc"),
    }
    label, cls = badges.get(tool_used, ("💭 General Knowledge", "badge-none"))
    return f'<span class="tool-badge {cls}">{label}</span>'

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("tool_used") is not None:
            st.markdown(tool_badge_html(msg["tool_used"]), unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg.get("tool_used") == "search_documents" and msg.get("sources"):
            with st.expander("📚 View source chunks"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(src[:500] + ("..." if len(src) > 500 else ""))

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, tool_used, sources = ask_agent(prompt)
            st.markdown(tool_badge_html(tool_used), unsafe_allow_html=True)
            st.markdown(answer)
            if tool_used == "search_documents" and sources:
                with st.expander("📚 View source chunks"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(src[:500] + ("..." if len(src) > 500 else ""))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tool_used": tool_used,
        "sources": sources
    })