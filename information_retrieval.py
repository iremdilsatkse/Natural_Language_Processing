from transformers import BertTokenizer, BertModel

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

documents = [
    "Machine learning is a field of artificial intelligence that focuses on learning from data",
    "Deep learning uses neural networks with many layers to model complex patterns",
    "I am watching TV series.",
    "Big data technologies allow us to process and analyze massive datasets",
    "Python is one of the most widely used languages in data science",
    "Neural networks are inspired by the structure of the human brain"
    
]

query = "What is machine learning?"

def get_embedding(text):
    inputs = tokenizer(text, return_tensors = "pt", truncations=True, padding = True)

    outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state

    embedding = last_hidden_state.mean(dim=1)

    return embedding.detach().numpy()


doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])
query_embedding = get_embedding(query)

similarity = cosine_similarity(query_embedding, doc_embeddings)

for i, score in enumerate(similarity[0]):
    print(f"Document: {documents[i]} \n{score}")

"""
Document: Machine learning is a field of artificial intelligence that focuses on learning from data
0.6905028820037842
Document: Deep learning uses neural networks with many layers to model complex patterns    
0.6601822972297668
Document: I am watching TV series.
0.6182225942611694
Document: Big data technologies allow us to process and analyze massive datasets
0.635374903678894
Document: Python is one of the most widely used languages in data science
0.6134124994277954
Document: Neural networks are inspired by the structure of the human brain
0.7069189548492432
"""








































