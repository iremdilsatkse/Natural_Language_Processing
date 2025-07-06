# Transformers NLP alanında önemli bir yere sahiptir. 
# Transformers, dil modelleme, metin sınıflandırma, soru cevaplama gibi birçok NLP görevinde kullanılabilir.

from transformers import AutoTokenizer, AutoModel
import torch

# model ve tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_name = AutoModel.from_pretrained(model_name)

# metni tanımla
text = "Transformers are a type of neural network architecture that has revolutionized natural language processing."

# metni tokenlara çevir
inputs = tokenizer(text, return_tensors="pt") # PyTorch tensörleri olarak döndürür

# modeli kullanarak metin temsili oluştur
with torch.no_grad():
    outputs = model_name(**inputs)

# modeliin çıktıısndan son gizli durumu al
last_hidden_state = outputs.last_hidden_state 

# ilk tokenın embeddingini al ve görüntüle
first_token_embedding = last_hidden_state[0, 0, :].numpy() 
print("Metin Temsili (embedding) ilk token için:")
print(first_token_embedding)