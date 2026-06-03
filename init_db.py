import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(DATA_DIR, "chroma_db")
TXT_PATH = os.path.join(DATA_DIR, "tagged_description.txt")

print("🚀 Starting optimized Vector Database initialization...")

# 1. Load and split documents
if not os.path.exists(TXT_PATH):
    raise FileNotFoundError(f"Could not find your text file at {TXT_PATH}")

print("📖 Reading tagged_description.txt...")
loader = TextLoader(TXT_PATH, encoding="utf-8")
raw_documents = loader.load()

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
print(f"📋 Found {len(documents)} total book records to embed.")

# 2. Load embedding model
print("🧠 Loading HuggingFace embedding model into memory...")
embedding_model = HuggingFaceEmbeddings()

# 3. Build database in optimized batches
print("⚡ Computing embeddings and saving to disk (Processing in batches)...")

# Clear out any old broken/half-finished databases first
if os.path.exists(DB_DIR):
    import shutil
    shutil.rmtree(DB_DIR)

# We process chunks of 200 documents at a time so your CPU doesn't stall
batch_size = 200
db = None

for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    print(f" -> Processing items {i} to {min(i + batch_size, len(documents))}...")
    
    if db is None:
        db = Chroma.from_documents(
            documents=batch,
            embedding=embedding_model,
            persist_directory=DB_DIR
        )
    else:
        db.add_documents(documents=batch)

print(f"✅ Success! Vector database built and saved to: {DB_DIR}")