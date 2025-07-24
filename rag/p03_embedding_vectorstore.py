# 1. Setup and Imports
import os
import warnings
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 2. Optional: Clean Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 3. Load environment variables (if needed)
load_dotenv()

# 4. Define Paths
DATA_PATH = "data/cleaned/wiki_medical_terms_cleaned.csv"
VECTORSTORE_PATH = "data/vectorstore/chroma_db"

# 5. Load Cleaned Dataset
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df)} records from cleaned dataset.")

# 6. Convert to LangChain Documents
documents = [
    Document(
        page_content=row["description"],
        metadata={
            "medical_term": row["medical_term"],
            "source": os.path.basename(DATA_PATH),
            "row_index": idx
        }
    )
    for idx, row in df.iterrows()
]
print(f"✅ Converted to {len(documents)} LangChain documents.")

# 7. Split into Chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=768,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""],
    length_function=len
)
docs_split = splitter.split_documents(documents)
print(f"✅ Split into {len(docs_split)} document chunks.")

# 8. Load Embedding Model
model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
embedding_model = HuggingFaceEmbeddings(model_name=model_name)
print(f"✅ Loaded embedding model: {model_name}")

# 9. Initialize Chroma Vectorstore
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory=VECTORSTORE_PATH
)

# 10. Embed in Batches with Horizontal Progress Bar
batch_size = 500
print(f"🔄 Embedding and storing {len(docs_split)} chunks to vectorstore...")

progress_bar = tqdm(
    range(0, len(docs_split), batch_size),
    desc="🔁 Building Vectorstore",
    dynamic_ncols=True,
    leave=False
)

for i in progress_bar:
    batch = docs_split[i:i + batch_size]
    vectorstore.add_documents(batch)

# 11. Persist Vectorstore
vectorstore.persist()
print(f"✅ Vectorstore created and persisted at: {VECTORSTORE_PATH}")