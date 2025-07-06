# Bag of Words nlp'de kullanılan bir tekniktir. Metinleri sayısal verilere dönüştürmek için kullanılır.
# Her kelime bir özellik olarak kabul edilir ve metinler bu özelliklerin varlığına veya yokluğuna göre temsil edilir.

from sklearn.feature_extraction.text import CountVectorizer

# Veri seti oluşturma
documents = [
    "Cat sat on the mat.",
    "Cat sat on the log."]

# Vectorizer tanımlama
vectorizer = CountVectorizer()

# Metni sayısal vektöre çevirme
X = vectorizer.fit_transform(documents)

# Kelime kümesi oluşturma
feature_names = vectorizer.get_feature_names_out() # kelime isimleri

# Vektör temsili
vector_representation = X.toarray()  

# Sonuçları yazdırma
print("Özellik İsimleri (Kelimeler):")
print(feature_names)
print("\nVektör Temsili:")
print(vector_representation)