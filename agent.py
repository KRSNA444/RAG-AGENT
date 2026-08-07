import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from rag_engine import RAGEngine

load_dotenv()

# --- Setup ---
rag = RAGEngine()
print("Indexing documents...")
rag.build_index("data")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=2
)
# --- Tool 1: RAG retrieval ---
@tool
def search_documents(query: str) -> str:
    """ALWAYS use this tool FIRST for any question about HTML, CSS, Bootstrap, JavaScript, or jQuery topics.
    This searches the user's uploaded PDF documents which contain the authoritative source material.
    Only use web_search if this tool returns no relevant information."""
    chunks = rag.retrieve(query, top_k=2)
    if not chunks:
        return "No relevant information found in documents."
    return "\n\n".join(chunks)

# --- Tool 2: Web search ---
web_search = TavilySearchResults(max_results=3)
web_search.name = "web_search"
web_search.description = "Search the internet for current, real-time, or general knowledge information not found in the user's documents."

# --- Tool 3: Calculator ---
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result.
    Input should be a valid Python math expression, e.g. '2 + 2' or '15 * 3.5'.
    Always state the final numeric answer clearly in your response."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"

# --- Create Agent ---
SYSTEM_PROMPT = """You are a helpful assistant with access to the user's uploaded documents.
When answering questions related to topics that might be in the user's documents (like HTML, CSS, Bootstrap, JavaScript, jQuery),
ALWAYS call search_documents FIRST and base your answer STRICTLY on the retrieved content.
Do NOT use your own general knowledge to answer if search_documents returns relevant content.
If search_documents returns no relevant info, then you may use web_search or your own knowledge, but mention that it's not from the user's documents.
When using the calculator tool, always state the exact numeric result clearly in your final answer."""

tools = [search_documents, web_search, calculator]
agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

def ask_agent(query, retries=2):
    for attempt in range(retries):
        try:
            result = agent.invoke({"messages": [("user", query)]})
            return result["messages"][-1].content
        except Exception as e:
            if "tool_use_failed" in str(e) and attempt < retries - 1:
                print("⚠️ Tool call failed, retrying...")
                time.sleep(1)
                continue
            return f"Error occurred: {e}"
# --- Main loop ---
if __name__ == "__main__":
    print("Agentic RAG ready! Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        answer = ask_agent(q)
        print(f"\nAgent: {answer}\n")