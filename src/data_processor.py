import pandas as pd
from pathlib import Path

def load_data():
    dataset_path = "hf://datasets/KadamParth/Ncert_dataset/NCERT_Dataset.csv"
    print(f"Loading dataset from Hugging Face: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise ConnectionError(f"Failed to load dataset. Ensure you are logged in via 'huggingface-cli login'. Error: {e}")

    df.columns = df.columns.str.strip()
    
    explanation_col = 'Explanation'
    question_col = 'Question'
    answer_col = 'Answer'
    
    if explanation_col not in df.columns or question_col not in df.columns or answer_col not in df.columns:
        raise ValueError(f"Dataset is missing required columns: '{explanation_col}', '{question_col}', or '{answer_col}'.")
        
    df['FullQuestion'] = df[explanation_col].fillna('') + " " + df[question_col].fillna('')
    
    df.dropna(subset=['FullQuestion', 'Answer'], inplace=True)

    # This now includes the Explanation for a richer context.
    df['qa_combined'] = "Explanation: " + df[explanation_col].fillna('') + \
                         "; Question: " + df['FullQuestion'] + \
                         "; Answer: " + df[answer_col].fillna('')
    
    df.rename(columns={'FullQuestion': 'Explanation Question'}, inplace=True)

    return df
