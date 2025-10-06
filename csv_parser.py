import pandas as pd

def extract_text_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string(index=False)