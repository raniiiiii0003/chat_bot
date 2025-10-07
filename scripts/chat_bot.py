import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.prompts import PromptTemplate


def start_chat_bot():
    # Paths
    DB_DIR = "./vector_store"

    # 1. Load embeddings (must match what we used in ingest.py)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Load FAISS vector store
    vectorstore = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    # retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Load local model (CPU friendly)
    # model_name = "google/flan-t5-large" # Text2Text generation model but week to fetch data
    # model_name = "openchat/openchat_3.5" # Big model
    model_name = "tiiuae/falcon-7b-instruct"  # Works
    llm_pipeline = pipeline("text-generation", model=model_name, device=0)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    prompt_template = """
    You are a helpful assistant that answers based ONLY on the context provided.
    
    Context:
    {context}
    
    Question:
    {question}
    
    If the answer is not in the context, reply with:
    "I don’t know based on the provided documents."
    
    Answer:
    """
    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # 4. Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    # 5. Chat loop
    print("Chatbot ready! Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        elif query.lower() in ["hi", "hello", "hey"]:
            print(f"Bot : Hi, how can i help you?")
            continue
        answer = qa.run(query)

        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    start_chat_bot()
