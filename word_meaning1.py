import nltk
from nltk.wsd import lesk

nltk.download("wordnet")
nltk.download("own-1.4")
nltk.download("punkt")

text1 = "I go to the bank to deposite money"
word1 = "bank"

sense1 = lesk(nltk.word_tokenize(text1), word1)
print("\nCümle:", text1)
print("Kelime:", word1)
print("Sense:", sense1.definition())

text2 = "The river bank is flooded after the heavy rain"
word2 = "bank"

sense1 = lesk(nltk.word_tokenize(text2), word2)
print("\nCümle:", text2)
print("Kelime:", word2)
print("Sense:", sense1.definition())