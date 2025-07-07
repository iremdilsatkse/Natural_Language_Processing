"""
cümleden duygu analizi
"""
from nltk.classify import MaxentClassifier

### Veri seti tanımlama
train_data = [
    ({"love": True, "amazing": True, "hate": False}, "positive"),
    ({"hate": True, "terrible": True}, "negative"),
    ({"joy": True, "happy": True, "hate": False}, "positive"),
    ({"sad": True, "depressed": True, "love": False}, "negative"),
    ({"excited": True, "happy": True}, "positive")]

### Max Entropy Sınıflandırıcı Eğitimi
classifier = MaxentClassifier.train(train_data, max_iter=10)

test_sentence = "I love this movie"
features = {word: (word in test_sentence.lower()) for word in ["love", "hate", "amazing", "terrible", "joy", "happy", "sad", "depressed", "excited"]}

label = classifier.classify(features)
print(f"Test cümlesi: '{test_sentence}' -> Duygu: {label}")