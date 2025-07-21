import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
import pickle
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent
INDEX_PATH = ROOT_DIR / 'saved_index' / 'ncert_faiss.index'
DATAFRAME_PATH = ROOT_DIR / 'saved_index' / 'data.pkl'

class RAGSystem:
    def __init__(self):
        print("--- Initializing RAG System ---")
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found. Please run src/embedding_builder.py")
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(INDEX_PATH))

        if not DATAFRAME_PATH.exists():
            raise FileNotFoundError(f"Dataframe pickle not found. Please run src/embedding_builder.py")
        print("Loading dataframe...")
        with open(DATAFRAME_PATH, 'rb') as f:
            self.df = pickle.load(f)
        
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file.")
        print("Initializing Grok client...")
        self.client = Groq(api_key=api_key)

        print("--- System is ready. ---")


    def retrieve_context(self, query: str, k: int = 3):
        query_vector = self.embedding_model.encode([query])
        _distances, indices = self.index.search(np.array(query_vector), k)
        retrieved_docs = [self.df['qa_combined'].iloc[i] for i in indices[0]]
        return retrieved_docs

    def get_answer(self, query: str):
        context_docs = self.retrieve_context(query)
        context_str = "\n\n".join(context_docs)

        system_prompt = f"""
        You are a helpful physics tutor. Your task is to answer the user's question based ONLY on the context provided below.
        The context contains a background explanation, a specific question, and a direct answer.

        First, provide the direct answer.
        Then, provide the broader explanation from the context to give more background.
        Structure your response clearly with "Answer:" and "Explanation:" headings.
        If the context does not contain the answer, just say "Sorry, I can't answer that based on the provided text."

        CONTEXT:
        {context_str}
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
                model="llama3-8b-8192",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Something went wrong with the Grok API call: {e}"

rag_system = RAGSystem()
