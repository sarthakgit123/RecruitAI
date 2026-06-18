import pickle

with open("faiss_db/metadata.pkl", "rb") as f:
    data = pickle.load(f)

print(data[0])