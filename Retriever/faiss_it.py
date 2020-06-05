import faiss
import sqlite3
import numpy as np
import pickle

# LOAD DATABASE
conn = sqlite3.connect('Database/reddit.db')
c = conn.cursor()


queries = []
idx = []

for row in c.execute('SELECT * FROM queries'):
    query_embd = np.frombuffer(row[-1], dtype=np.float32)
    queries.append(query_embd)
    idx.append(str(row[0]))

queries = np.asarray(queries).reshape((-1, 1024))

index = faiss.IndexIDMap(faiss.IndexFlatIP(1024))
index.add_with_ids(queries, np.array(range(0, len(idx))))
print(index.is_trained)
print(queries[0:5])
print("\n\n")
print(index.search(queries[0:5],k=5))


faiss.write_index(index, "Faiss_index/large.index")

with open("Faiss_index/thread_idx.pkl",'wb') as fp:
    pickle.dump(idx,fp)

