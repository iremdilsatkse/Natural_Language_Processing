"""
spam veri setinin içerisinde bulunan spam ve ham verileri binary classification ile sınıflandıracağız
"""
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

data = pd.read_csv("spam.csv", encoding="Latin-1")
# print(data)
# print(data.columns)
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
data.columns = ["Label", "Text"]

### EDA (kayıp veri kontolü)
# print(data.isna().sum())

### Veri Temizleme ve Preprocessing
nltk.download("stopwords") # anlam taşımayan sözcükler
nltk.download("wordnet") # lemma bulmak için 
nltk.download("omw-1.4") # wordnete ait farklı dillerin kelimeleri

text = list(data.Text)
lemmatizer = WordNetLemmatizer()

corpus = []

for i in range(len(text)):
    r = re.sub("[a-zA-Z]", " ", text[i])
    r = r.lower()
    r = r.split()
    r = [word for word in r if word not in stopwords.words("english")]
    r = [lemmatizer.lemmatize(word) for word in r]
    r = " ".join(r)
    corpus.append(r)

data["text2"] = corpus

### Model Eğitimi ve Değerlendirme
X = data["text2"]
y = data["Label"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

# Feature Extract
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)

dt = DecisionTreeClassifier()
dt.fit(X_train_cv, y_train)

X_test_cv = cv.transform(X_test)

### Prediction
prediction = dt.predict(X_test_cv)

c_matrix = confusion_matrix(y_test, prediction)
print(c_matrix)

accuracy = 100 * (sum(sum(c_matrix)) - c_matrix[1,0] - c_matrix[0,1]) / sum(sum(c_matrix))
print(accuracy)