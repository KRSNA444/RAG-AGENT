import os
from dotenv import load_dotenv
from groq import Groq
from rag_engine import RAGEngine

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
rag = RAGEngine()

print("📚 Indexing documents...")
rag.build_index("data")

def ask(query):
    retrieved_chunks = rag.retrieve(query, top_k=3)
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the provided documents."
When answering, please explicitly cite the Source and Page number from the context where you found the information.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    print("🤖 RAG Agent ready! Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        answer = ask(q)
        print(f"\nAgent: {answer}\n")