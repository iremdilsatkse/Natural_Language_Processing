import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter

import nltk
from nltk.corpus import stopwords

data = pd.read_csv("IMDB Dataset.csv")
# print(data.head())
nltk.download('stopwords')

documents = data['review']
labels = data['sentiment'] # pozitif veya negatif

### Veri temizleme
stopwords.words('english')  
english_stopwords = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  # Büyük/Küçük harf çevrimi
    text = re.sub(r"\d+", " ", text)  # Sayıların kaldırılması
    text = re.sub(r"[^\w\s]", " ", text)  # Özel işaretlerin kaldırılması
    text = " ".join([word for word in text.split() if len(word) > 2])  # 2 karakterden kısa kelimelerin kaldırılması
    return text

def remove_stopwords(text, stopwords):
    return " ".join([word for word in text.split() if word not in stopwords])

cleaned_documents = [clean_text(row) for row in documents]
cleaned_documents_no_stopwords = [remove_stopwords(doc, english_stopwords) for doc in cleaned_documents]


### BOW (Bag of Words) 
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(cleaned_documents_no_stopwords[:50])

# Kelime kümesinin gösterilmesi
feature_names = vectorizer.get_feature_names_out()

# Vektör temsilinin gösterilmesi
print("Vektör Temsili:")
print(X.toarray())

df_bow = pd.DataFrame(X.toarray(), columns=feature_names)
print("\nBOW DataFrame:")
print(df_bow.head())

# Kelime frekanslarının gösterilmesi
word_counts = X.sum(axis=0).A1  
word_freq = dict(zip(feature_names, word_counts))

print("\nKelime Frekansları:")
for word, freq in Counter(word_freq).most_common(10):
    print(f"{word}: {freq}")

