import spacy

nlp = spacy.load("en_core_web_sm")

word = "They love eating chocolate"

doc = nlp(word)

for token in doc:
    print("\nText:", token.text) # kelime
    print("Lemma:", token.lemma_) # kelimenin kökü
    print("POS:", token.pos_) # kelimenin türü
    print("Tag:", token.tag_) # kelimenin dil bilgisi özelliği
    print("Dependency:", token.dep_) # kelimenin rolü
    print("Shape:", token.shape_) # kelimenin karakter yapısı
    print("Is alpha:", token.is_alpha) # kelimenin yalnızca alfabetik karakterlerden oluşup oluşmadığını kontrol eder
    print("Is stop:", token.is_stop) # kelime stop words mü değil mi
    print("Morphology:", token.morph) # kelimenin morfolojik özellikleri














