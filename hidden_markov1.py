# Hidden Markov modeli bir dizi gözlemin arkasında gizli bir durum dizisinin olduğu varsayımına dayanır.
"""
Part of Speech POS: kelimelerin uygun sözcük tüürnü bulma çalışması
"""
import nltk
from nltk.tag import hmm

train_data = [
    [("I", "PRP"), ("am", "VBP"), ("happy", "JJ"), ("with", "IN"), ("you", "PRP")],
    [("She", "PRP"), ("loves", "VBZ"), ("to", "TO"), ("read", "VB"), ("books", "NNS")],
    [("They", "PRP"), ("love", "VBP"), ("to", "TO"), ("play", "VB"), ("football", "NN")],
    [("He", "PRP"), ("loves", "VBZ"), ("to", "TO"), ("watch", "VB"), ("movies", "NNS")]
]

# HMM modelini eğitme
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

test_sentence = "I love to learn new things".split()

tags = hmm_tagger.tag(test_sentence)
print("Tagged Sentence:", tags)

test_sentence2 = "He is a driver".split()

tags = hmm_tagger.tag(test_sentence2)
print("Tagged Sentence:", tags)
