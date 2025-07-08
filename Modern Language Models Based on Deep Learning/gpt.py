from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"

# tokenization ve model oluşturma
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model = GPT2LMHeadModel.from_pretrained(model_name)

# text
text = "Afternoon,"

inputs = tokenizer.encode(text, return_tensors="pt")
outputs = model.generate(inputs, max_length=50)

generated_text =tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)