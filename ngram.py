# Dil modelinde kullanılan kelime veya karakter dizisinin uzunluğunu belirten bir terimdir.
# N-gram, bir dil modelinde ardışık n öğeden oluşan bir dizidir.

from sklearn.feature_extraction.text import CountVectorizer

# Örnek metin
documents = [
    "Hello world",
    "This is a natural language processing example"]

# Unigram, Bigram ve Trigram oluşturma
vectorizer_unigram = CountVectorizer(ngram_range=(1, 1))
vectorizer_bigram = CountVectorizer(ngram_range=(2, 2))
vectorizer_trigram = CountVectorizer(ngram_range=(3, 3))

X_unigram = vectorizer_unigram.fit_transform(documents)
unigram_features = vectorizer_unigram.get_feature_names_out()

X_bigram = vectorizer_bigram.fit_transform(documents)
bigram_features = vectorizer_bigram.get_feature_names_out()

X_trigram = vectorizer_trigram.fit_transform(documents)
trigram_features = vectorizer_trigram.get_feature_names_out()

# Sonuçları yazdırma
print("\nUnigram Features:", unigram_features)
print("\nBigram Features:", bigram_features)
print("\nTrigram Features:", trigram_features)