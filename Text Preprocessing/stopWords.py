# Stop words, metinlerde sıkça kullanılan ancak anlamı olmayan kelimelerdir.
# Bu kelimeler, metin madenciliği ve doğal dil işleme (NLP) uygulamalarında genellikle filtrelenir.
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

### İngilizce stop words listesi
stopwords.words('english')  
english_stopwords = set(stopwords.words('english'))

# print("İngilizce Stop Words:")
# print(english_stopwords)

text = "This is a sample sentence demonstrating the use of stop words in natural language processing."

text_list = text.split()
filtered_words = [word for word in text_list if word.lower() not in english_stopwords]
print("\nFiltrelenmiş Kelimeler (İngilizce):")
print(filtered_words)

### Türkçe stop words analizi
turkish_stopwords = set(stopwords.words('turkish'))

# print("\nTürkçe Stop Words:")
# print(turkish_stopwords)

text2 = "Bu doğal dil işleme uygulamalarında durdurma kelimelerinin kullanımını gösteren örnek bir cümledir."

text_list2 = text2.split()
filtered_words2 = [word for word in text_list2 if word.lower() not in turkish_stopwords]
print("\nFiltrelenmiş Kelimeler (Türkçe):")
print(filtered_words2)

### Kütüphanesiz stop words analizi
our_stopwords = ["için", "ve", "ama", "ben"]

text3 = "Bu metin için örnek bir cümledir ve ben durdurma kelimelerinin kullanımını gösteriyorum."

text_list3 = text3.split()
filtered_words3 = [word for word in text_list3 if word.lower() not in our_stopwords]
print("\nKütüphanesiz Filtrelenmiş Kelimeler:")
print(filtered_words3)


