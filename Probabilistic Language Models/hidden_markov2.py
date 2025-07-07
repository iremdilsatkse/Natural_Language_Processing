import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000

### Veri setini içeri aktarma
nltk.download('conll2000')

train_data = conll2000.tagged_sents('train.txt')
test_data = conll2000.tagged_sents('test.txt')  

# print("Train data:", train_data[:5])

### HMM modelini eğitme
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# Test cümlesi
test_sentence = "The quick brown fox jumps over the lazy dog".split()
tags = hmm_tagger.tag(test_sentence)
print("Tagged Sentence:", tags)
