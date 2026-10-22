import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

print("=== Starting Overnight Book Embedding ===")

# 1. Verify books directory exists
books_dir = "./books"
if not os.path.exists(books_dir):
    os.makedirs(books_dir)
    print(f"Created {books_dir}/ directory. Please place your PDF files inside it.")

# 2. Load all PDF files from ./books
print("\n[1/3] Scanning and loading PDF files from ./books ...")
loader = DirectoryLoader(books_dir, glob="*.pdf", loader_cls=PyPDFLoader, show_progress=True)
documents = loader.load()

if not documents:
    print("No PDF files found in ./books. Please add your textbooks and try again.")
    exit()

print(f"-> Successfully loaded {len(documents)} total pages across all books.")

# 3. Split documents into chunks
print("\n[2/3] Chunking text for vector processing ...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print(f"-> Created {len(chunks)} text chunks.")

# 4. Generate vector embeddings and persist to disk
print("\n[3/3] Generating embeddings with nomic-embed-text ...")
print("This may take 30–60 minutes depending on total page count. You can leave this running overnight.")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("\n=== COMPLETE! ===")
print("All textbooks have been embedded and saved into './chroma_db'.")
