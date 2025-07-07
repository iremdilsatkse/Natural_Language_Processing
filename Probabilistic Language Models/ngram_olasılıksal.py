"""
Bu dosyada dil modeli oluşturulacak. 
Bir kelimdeden sonra gelebilecek kelimelerin olasılıklarını hesaplamak için n-gram model kullanılacak.
"""

import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from collections import Counter

### Veri seti oluşturma
corpus = [
    "I love programming in Python.",
    "She loves to read books.",
    "I love to learn new things.",
    "We love to explore new technologies.",
    "They love to play football.",
    "I love to travel around the world."]

### Verileri tokenize et
tokens = [word_tokenize(sentence.lower()) for sentence in corpus]
# print("Tokens:", tokens)

### Bigram oluşturma
bigrams = []
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list, 2)))

bigrams_frequency = Counter(bigrams)

# print("Bigrams:", bigrams)
# print("Bigrams Frequency:", bigrams_frequency)

### Trigram oluşturma
trigrams = []
for token_list in tokens:
    trigrams.extend(list(ngrams(token_list, 3)))

trigrams_frequency = Counter(trigrams)

# print("Trigrams:", trigrams)
# print("Trigrams Frequency:", trigrams_frequency)

### Modeli test etme
"""
"I love" bigramından sonra "programming" veya "to" kelimelerinin olasılıklarının hesaplanması
"""

bigram = ('i', 'love')

# "i love programming" olma olasılığı
prob_programming = trigrams_frequency[("i", "love", "programming")] / bigrams_frequency[bigram] 

print("Programming olasılığı:", prob_programming)

# "i love to" olasılığı
prob_to = trigrams_frequency[("i", "love", "to")] / bigrams_frequency[bigram]

print("To olasılığı:", prob_to)