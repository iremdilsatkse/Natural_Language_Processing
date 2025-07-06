# TF-IDF (Term Frequency-Inverse Document Frequency) 
# Kelimelerin belgelerdeki önemini ölçen bir istatistiksel ölçüdür.

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Örnek belge oluşturma
documents = [
    "Python is a programming language.",
    "Python is widely used in data science.",
    "Data science involves statistics and machine learning.",
    "Machine learning is a subset of artificial intelligence."]

# Vektörizer tanımlama
tfidf_vectorizer = TfidfVectorizer()

# Sayısal hale çevirme
X = tfidf_vectorizer.fit_transform(documents)

# Kelime kümesini inceleme
feature_names = tfidf_vectorizer.get_feature_names_out()

# Vektör temsilini inceleme
vector_representation = X.toarray()

df_tfidf = pd.DataFrame(vector_representation, columns=feature_names)
print("\nTF-IDF DataFrame:")
print(df_tfidf)

# Ortalama tf-idf değerlerine bakma
tf_idf = df_tfidf.mean(axis=0)
print("\nAverage TF-IDF values:")
print(tf_idf.sort_values(ascending=False))

