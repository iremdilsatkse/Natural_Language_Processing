"""
Varlık İsmi Tanıma (NER)
Metin içerisindeki özel isimleri bulmaya yarar.
Yer, isim, tarih vb.
"""
import pandas as pd
import spacy

spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

text = "Dylan O'Brien is an actor who plays Stiles in the TV series Teen Wolf. The series is set in Beacon Hills, California."

doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
    # ent.text: varlık ismi
    # ent.start_char: varlığın metindeki başlangıç karakteri
    # ent.end_char: varlığın metindeki bitiş karakteri
    # ent.label_: varlık türü

entities = [(ent.text, ent.label_, ent.lemma_)for ent in doc.ents]

df = pd.DataFrame(entities, columns=["text", "type", "lemma"])
print(df)








































