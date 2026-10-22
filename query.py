import sys
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load the existing local vector database
print("Loading vector database...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)
retriever = db.as_retriever(search_kwargs={"k": 3})

# 2. Connect to local offline LLM
llm = OllamaLLM(model="llama3.2:1b")

# 3. Create RAG prompt & pipeline
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Process terminal query
if len(sys.argv) > 1:
    user_query = " ".join(sys.argv[1:])
    print(f"\nQuerying: {user_query}\n" + "-"*40)
    response = rag_chain.invoke(user_query)
    print("\nAnswer:\n", response)
else:
    print("Please provide a question. Example: python3 query.py 'What is WACC?'")
