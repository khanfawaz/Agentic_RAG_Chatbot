import requests
from pathlib import Path

# Constants
DATA_URL = "https://huggingface.co/datasets/gamino/wiki_medical_terms/resolve/main/wiki_medical_terms.parquet"
RAW_DIR = Path("data/raw")
RAW_FILE = RAW_DIR / "wiki_medical_terms.parquet"

# Ensure raw directory exists
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Download only if file doesn't exist
if not RAW_FILE.exists():
    print("⬇️ Downloading dataset...")
    response = requests.get(DATA_URL)
    if response.status_code == 200:
        with open(RAW_FILE, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded to: {RAW_FILE}")
    else:
        raise Exception(f"❌ Download failed. Status code: {response.status_code}")
else:
    print(f"✅ Dataset already exists at: {RAW_FILE}")