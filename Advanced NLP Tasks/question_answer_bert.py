from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import warnings
warnings.filterwarnings("ignore")

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

tokenizer = BertTokenizer.from_pretrained(model_name)

# Soru cevaplama görevi için BERT modeli
model = BertForQuestionAnswering.from_pretrained(model_name)

# Cevapları tahmin eden fonksiyon
def answer(context, question):
    """
        context: metin
        question: soru
        Amaç: metin içerisinden soruyu bulma
    """

    # metnin token haline getirilmesi
    encoding = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
    
    # giriş tensörleri
    input_ids = encoding["input_ids"] # tokenların id
    attention_mask = encoding["attention_mask"] # hangi tokenların dikkate alınacağı
    
    with torch.no_grad():
        start_scores, end_scores = model(input_ids, attention_mask = attention_mask, return_dict = False)

    # en yüksek olasılığa sahip yer
    start_index = torch.argmax(start_scores, dim=1).item()
    end_index = torch.argmax(end_scores, dim=1).item()

    # cevabın elde edilmesi
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_index: end_index + 1])

    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer

question = "What is the capital of Turkey"
context = "Surrounded by the fertile lands of Anatolia, Ankara, the capital of modern Türkiye, has been home to many civilizations over the centuries, including the Hittites, the most powerful civilizations of antiquity, as well as the Phrygians, Galatians and Romans."

answer = answer(context, question)
print("Answer:", answer)


























