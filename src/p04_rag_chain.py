# p04_rag_with_vectorstore.py

from dotenv import load_dotenv
load_dotenv()


import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# --- Safety check ---
assert os.getenv("GROQ_API_KEY"), "❌ GROQ_API_KEY is not set"
print("✅ GROQ_API_KEY is set")

# ---- Step 1: Load Embedding Model ----
model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
embedding_model = HuggingFaceEmbeddings(model_name=model_name)
print(f"✅ Loaded embedding model: {model_name}")

# ---- Step 2: Load Vectorstore from Disk ----
VECTORSTORE_PATH = "data/vectorstore/chroma_db"
vectorstore = Chroma(
    persist_directory=VECTORSTORE_PATH,
    embedding_function=embedding_model
)
print(f"✅ Loaded Chroma vectorstore from: {VECTORSTORE_PATH}")

# ---- Step 3: Configure Retriever ----
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print(f"✅ Retriever initialized with top-k = 4")

# ---- Step 4: Load LLM from Groq ----
llm = ChatGroq(
    temperature=0.2,
    model_name="llama-3.3-70b-versatile"
)
print("✅ Groq LLM (LLaMA3-70B) initialized.")

# ---- Step 5: Define Prompt Template ----
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful and knowledgeable medical assistant. Use the following context to answer the user's question.
If you're unsure or the answer isn't in the context, just say "I don't know" — don't make things up.

Context:
{context}

Question:
{question}

Answer:
"""
)

# ---- Step 6: Create RAG QA Chain ----
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)
print("✅ RAG QA chain initialized and ready.")

# ---- Step 7: Interactive Q&A Loop ----
while True:
    user_query = input("\n🔍 Ask a medical question (or type 'exit'): ")
    if user_query.lower() == "exit":
        print("👋 Exiting RAG assistant.")
        break

    result = qa_chain.invoke({"query": user_query})
    print("\n✅ Answer:\n", result["result"])

    print("\n📚 Sources:")
    for i, doc in enumerate(result["source_documents"], start=1):
        metadata = doc.metadata
        filename = metadata.get("source") or metadata.get("medical_term", "Unknown File")
        print(f"{i}. {filename}")