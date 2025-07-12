from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)

model = MarianMTModel.from_pretrained(model_name)

text = "Hi, What's your name"

translated_text = model.generate(**tokenizer(text, return_tensors = "pt", padding = True))

translated_text = tokenizer.decode(translated_text[0], skip_special_tokens = True)
print("Translated Text:", translated_text)







