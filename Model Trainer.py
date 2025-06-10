import pandas as pd
import faiss
import requests
import numpy as np
import chardet

# Opening The File
with open("small_dataset.csv", "rb") as f:
    result = chardet.detect(f.read())
encoding_detected = result["encoding"]
with open("small_dataset.csv", "r", encoding=encoding_detected, errors="replace") as f:
    df = pd.read_csv(f)

# A function to improve the textual representation of the database
def ctr(row):
    return f"""Type: {row['type']},
Title: {row['title']},
Year: {row['release_year']},
Duration: {row['duration']},
Description: {row['description']}"""
df['tr'] = df.apply(ctr, axis=1)

# Initialize FAISS index
dim = 3072
index = faiss.IndexFlatL2(dim)
X = np.zeros((len(df['tr']), dim), dtype='float32')

# Generate embeddings
for i, representation in enumerate(df['tr']):
    if i % 2 == 0:
        print('Processed', str(i), 'instances')

    res = requests.post("http://localhost:11434/api/embeddings",
        json={'model': 'llama3.2', 'prompt': representation}
    )

    embedding = res.json().get('embedding', [])  # Handle missing embeddings safely
    if embedding:
        X[i] = np.array(embedding, dtype='float32')

index.add(X)
faiss.write_index(index, "index")
print("Index saved successfully!")
