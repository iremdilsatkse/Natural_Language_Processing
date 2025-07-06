import nltk

# metni cümle ve kelime bazında tokenlara ayırmak için
nltk.download('punkt')
nltk.download('punkt_tab')

text = "This is an example text. It contains multiple sentences. Hi ..."

# Kelime bazında tokenizasyon : Noktalama işaretleri ve boşlukları ayrı token olarak elde eder.
word_tokens = nltk.word_tokenize(text)
print("Kelime bazında tokenizasyon:")
print(word_tokens)

# Cümle bazında tokenizasyon : Cümleleri ayrı token olarak elde eder.
sentence_tokens = nltk.sent_tokenize(text)
print("\nCümle bazında tokenizasyon:")
print(sentence_tokens)