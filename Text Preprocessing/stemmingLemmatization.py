# Stemming işlemi ve lemmatization işlemi, kelimeleri köklerine indirgemek için kullanılır.
# Stemming, kelimenin kökünü bulmaya çalışırken, lemmatization ise kelimenin kökünü ve anlamını dikkate alır.

import nltk

nltk.download('wordnet') # lemmatization işlemi için gerekli veri tabanı

# Stemming işlemi
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["eating", "eats", "ate", "better", "go", "went"]

# Kelimeleri köklerine indirgeme işlemi
stems = [stemmer.stem(w) for w in words]

print("Stemming Sonuçları:")
print(stems)

# Lemmatization işlemi
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

lemmas = [lemmatizer.lemmatize(w, pos="v") for w in words] # kelimeleri fiil olarak işlemek için pos="v" kullanıldı

print("\nLemmatization Sonuçları:")
print(lemmas)