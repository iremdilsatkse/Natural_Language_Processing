import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re

import nltk
from nltk.corpus import stopwords

# veri seti yükle
data = pd.read_csv("C:/Users/İrem/OneDrive/Belgeler/VsCode/UcanbleBootcamp/spam.csv", encoding='latin1')

nltk.download('stopwords')

# veri temizleme
stopwords.words('english')  
english_stopwords = set(stopwords.words('english'))

def clean_text(v2):
    text = text.lower()  # Büyük/Küçük harf çevrimi
    text = re.sub(r"\d+", " ", text)  # Sayıların kaldırılması
    text = re.sub(r"[^\w\s]", " ", text)  # Özel işaretlerin kaldırılması
    text = " ".join([word for word in text.split() if len(word) > 2])  # 2 karakterden kısa kelimelerin kaldırılması
    return v2

def remove_stopwords(v2, english_stopwords):
    return " ".join([word for word in v2.split() if word not in english_stopwords])

### tfidf
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data.v2)

# kelime kümesini incele
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = X.mean(axis=0).A1  

#tfidf skorlarını içerem bir df oluştur
df_tfidf = pd.DataFrame({"word": feature_names, "tfidf_score": tfidf_scores})

# skorları sırlaara ve yazdır
df_tfidf = df_tfidf.sort_values(by='tfidf_score', ascending=False)
print("\nTF-IDF DataFrame:")
print(df_tfidf.head(10))

# Kelime frekanslarını göster
word_freq = dict(zip(feature_names, tfidf_scores))
print("\nKelime Frekansları:")
for word, freq in sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:10]:
    print(f"{word}: {freq:.4f}")    

