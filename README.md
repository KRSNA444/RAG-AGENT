# RAG Agent

An agentic Retrieval-Augmented Generation (RAG) system that answers questions from your own PDF documents and decides on its own when to search the documents, search the web, or run a calculation.

Built with LangGraph, FAISS, sentence-transformers, Groq LLMs and a Streamlit interface.

**Live demo:** _add your Streamlit link here_
**Author:** Krishnakanth ([GitHub](https://github.com/KRSNA444))

---

## Features

- **Document Q&A (RAG):** PDFs in the `data/` folder are parsed, chunked, embedded and indexed. The agent answers from the retrieved content first.
- **Agentic tool routing:** a LangGraph ReAct agent chooses between three tools:
  | Tool | Purpose |
  | --- | --- |
  | `search_documents` | Semantic search over your PDFs (FAISS) |
  | `web_search` | Live web search via Tavily when the documents have no answer |
  | `calculator` | Evaluates math expressions |
- **Document-first behaviour:** the agent is instructed to rely on retrieved content and to say when an answer is not from your documents.
- **Retry on tool-call failures** for more reliable responses.
- **Three ways to run it:** Streamlit web app (`app.py`), agent CLI (`agent.py`) and a simple RAG-only CLI (`main.py`).

## How it works

```
PDFs (data/)
   │  PyMuPDF text extraction
   ▼
RecursiveCharacterTextSplitter (500 chars, 50 overlap)
   │
   ▼
all-MiniLM-L6-v2 embeddings ──► FAISS index (IndexFlatL2)
                                      │
User question ──► LangGraph ReAct agent (Groq LLM)
                     ├── search_documents ──► top-k chunks
                     ├── web_search (Tavily)
                     └── calculator
                     │
                     ▼
                  Final answer
```

## Tech stack

- **LLM:** Groq (`openai/gpt-oss-20b` in the agent, `llama-3.3-70b-versatile` in the simple CLI)
- **Agent framework:** LangGraph, LangChain
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector store:** FAISS (CPU)
- **PDF parsing:** PyMuPDF
- **Web search:** Tavily
- **UI:** Streamlit

## Project structure

```
.
├── app.py             # Streamlit web app
├── agent.py           # LangGraph agent with 3 tools (CLI)
├── main.py            # Simple RAG-only CLI
├── rag_engine.py      # PDF loading, chunking, embedding, FAISS retrieval
├── data/              # Put your PDF files here
├── requirements.txt
└── .env.example
```

## Getting started

1. **Clone the repo**
   ```bash
   git clone https://github.com/KRSNA444/RAG-AGENT.git
   cd RAG-AGENT
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Add your API keys**

   Copy `.env.example` to `.env` and fill in:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```
   Free keys: [Groq Console](https://console.groq.com) and [Tavily](https://tavily.com).

4. **Add your documents**

   Place one or more `.pdf` files inside the `data/` folder.

5. **Run**
   ```bash
   streamlit run app.py     # web app
   python agent.py          # agent in the terminal
   python main.py           # simple RAG in the terminal
   ```

## Example questions

- _"Explain the difference between `let` and `var`."_ (answered from your documents)
- _"What is the latest version of Bootstrap?"_ (web search)
- _"What is 15 * 3.5 + 120?"_ (calculator)

## Screenshots

_Add 2-3 screenshots of the app here, e.g. `docs/screenshot-1.png`._

## Roadmap

- Store source file and page number with each chunk and show citations in answers
- Support uploading PDFs from the UI
- Add a small evaluation set to measure retrieval quality
- Add tests and CI

## License

_Add a license (e.g. MIT) if you want others to reuse the code._
