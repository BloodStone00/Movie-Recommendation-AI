import pandas as pd
import faiss
import requests
import numpy as np
import chardet


#Opening File
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


#Now taking a new movie finding its embedding and finiding the
#closest matches to that movie

reprasentation = '''Type: {movie},
Title: {Kraven the Hunter},
Year: {2024},
Duration: {127},
Description: {Kraven's complex relationship with his ruthless father starts him down a path of vengeance, motivating him to become not only the greatest hunter in the world, but also one of its most feared.}
'''


res = requests.post('http://localhost:11434/api/embeddings', json={
    'model': 'llama3.2',
    'prompt': reprasentation
})

#Searching for the nearest embeddings
index = faiss.read_index("index")
embedding = np.array(res.json()['embedding'], dtype='float32')
D, I = index.search(embedding.reshape(1, -1), 2)
best_matches = np.array(df['tr'])[I.flatten()]
for match in best_matches:
    print('NEXT MOVIE')
    print(match)
    print(' ')




