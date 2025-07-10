import spacy

nlp = spacy.load("en_core_web_sm")

sentence = "i walked around in my pajamas today, it was a comfortable day"
doc1 = nlp(sentence)

for token in doc1:
    print(token.text, token.pos_)

"""
i PRON
walked VERB
around ADV
in ADP
my PRON
pajamas NOUN
today NOUN
, PUNCT
it PRON
was AUX
a DET
comfortable ADJ
day NOUN
"""