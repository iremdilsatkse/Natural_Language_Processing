import pandas as pd
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

data = pd.read_csv("amazon.csv")

lemmatizer = WordNetLemmatizer()
def clean_data(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords.words("english")]
    lemmatized_token = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    processed_text = " ".join(lemmatized_token)
    return processed_text

data["reviewText2"] = data["reviewText"].apply(clean_data)

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    sentiment = 1 if score["pos"] > 0 else 0
    return sentiment

data["sentiment"] = data["reviewText2"].apply(get_sentiment)

##############
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(data["Positive"], data["sentiment"])
print("Confusion Matrix:", cm)

cr = classification_report(data["Positive"], data["sentiment"])
print("Classification Report:", cr)






