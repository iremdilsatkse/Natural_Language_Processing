from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import warnings
warnings.filterwarnings("ignore")


model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Soru cevaplama görevi için GPT2 modeli
model = GPT2LMHeadModel.from_pretrained(model_name)

# Cevapları tahmin eden fonksiyon
def answer(context, question):
    input_text = f"Question: {question}, Context: {context}. Please answer the question by analyzing the text provided. Just write the answer to the question."

    # metnin token haline getirilmesi
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(inputs, max_length = 300)

    answer = tokenizer.decode(outputs[0], skip_special_tokens = True)

    answer = answer.split("Answer:")[-1].strip()

    return answer

question = "What is the capital of Turkey"
context = "Surrounded by the fertile lands of Anatolia, Ankara, the capital of modern Türkiye, has been home to many civilizations over the centuries, including the Hittites, the most powerful civilizations of antiquity, as well as the Phrygians, Galatians and Romans."

answer = answer(context, question)
print("Answer:", answer)






















