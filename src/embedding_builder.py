import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
from data_processor import load_data

ROOT_DIR = Path(__file__).parent.parent
INDEX_PATH = ROOT_DIR / 'saved_index' / 'ncert_faiss.index'
DATAFRAME_PATH = ROOT_DIR / 'saved_index' / 'data.pkl'

INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

def build_and_save_index():
    print("Reading the data from Hugging Face...")
    df = load_data() # No longer needs a path
    questions = df['Explanation Question'].tolist()
    print("Loading the embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Creating vector embeddings...")
    question_embeddings = model.encode(questions, convert_to_tensor=False, show_progress_bar=True)
    d = question_embeddings.shape[1]
    print(f"Building the FAISS index...")
    index = faiss.IndexFlatL2(d)
    index.add(np.array(question_embeddings))
    print(f"Saving the index to disk...")
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Saving the dataframe...")
    with open(DATAFRAME_PATH, 'wb') as f:
        pickle.dump(df, f)
    print("\nAll done! Your index is ready.")

if __name__ == "__main__":
    build_and_save_index()
