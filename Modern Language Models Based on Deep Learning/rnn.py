"""
Duygu analizi için RNN modeli
Bir cümleninin etiketlenmesi (pozitif, negatif)

Müşteri yorumları üzerinde duygu analizi yapacağız.
"""
import numpy as np
import pandas as pd

from gensim.models import Word2Vec # metin temsili
from keras_preprocessing.sequence import pad_sequences # padding işlemi
from keras.models import Sequential # model oluşturma
from keras.layers import Embedding, SimpleRNN, Dense # katmanlar
from tensorflow.keras.preprocessing.text import Tokenizer # metin tokenizasyonu

from sklearn.preprocessing import LabelEncoder # etiket kodlama
from sklearn.model_selection import train_test_split # veri setini train ve test olarak ayırma

# veri seti oluşturma
data = {
    "text": [
        "Yemekler çok lezzetliydi, bayıldık!",
        "Servis çok yavaş ve ilgisizdi.",
        "Tatlılar enfesti, tekrar geleceğiz.",
        "Garsonlar kaba davranışlar sergiledi.",
        "Ortam çok huzurluydu, keyif aldık.",
        "Siparişimiz karıştı, memnun kalmadık.",
        "Sunum ve lezzet harikaydı.",
        "Çatal bıçaklar kirliydi, hijyen yetersiz.",
        "Pizza sıcak ve bol malzemeliydi.",
        "Kahve çok acıydı, içemedik.",
        "Çalışanlar çok güleryüzlüydü.",
        "Hesapta yanlışlık yaptılar.",
        "Tatlı porsiyonları yeterli ve tazeydi.",
        "Rezervasyon yaptırmamıza rağmen bekledik.",
        "Mekanın ambiyansı harikaydı.",
        "Yemek çok tuzluydu, yiyemedik.",
        "Çorba sıcaktı ve çok lezzetliydi.",
        "Garson siparişi unuttu.",
        "Müzik çok hoştu, keyifli bir ortam vardı.",
        "Fiyatlar fazla yüksekti.",
        "Sunum özenliydi, küçük detaylara dikkat edilmişti.",
        "Peçetelikler boştu, ilgisizlik vardı.",
        "Tatlar birbiriyle uyumluydu.",
        "İçerisi çok gürültülüydü.",
        "Garson çocuklara çok nazikti.",
        "Yemekte kıl vardı, iğrendiriciydi.",
        "Kahvaltı tabağı zengindi ve doyurucuydu.",
        "Limonata çok ekşiydi, içemedim.",
        "İçecekler soğuk ve tazeydi.",
        "Sandalyeler rahattı.",
        "Yemekler soğuk geldi.",
        "Servis çok hızlıydı.",
        "Tatlı çok bayattı.",
        "Manzara harikaydı, çok beğendik.",
        "Garson hiç yardımcı olmadı.",
        "Masalar temizdi, hijyene önem verilmiş.",
        "Yemeklerde baharat aşırıydı.",
        "Makarna al dente kıvamındaydı.",
        "Garson yanlış sipariş getirdi.",
        "Masamız önceden hazırlanmıştı, hoşumuza gitti.",
        "Hesap çok yüksekti.",
        "Tatlıyı ikram ettiler, çok naziktiler.",
        "Yemeklerde yeterince tuz yoktu.",
        "İlgili ve profesyonel bir ekip vardı.",
        "Yemekler geç geldi, çok bekledik.",
        "Girişte güler yüzle karşılandık.",
        "İçeride ağır bir yemek kokusu vardı.",
        "Sunum çok yaratıcıydı.",
        "İçecekler geç geldi, yemek bitmişti."
    ],
    "label": [
        "pozitif", "negatif", "pozitif", "negatif", "pozitif",
        "negatif", "pozitif", "negatif", "pozitif", "negatif",
        "pozitif", "negatif", "pozitif", "negatif", "pozitif",
        "negatif", "pozitif", "negatif", "pozitif", "negatif",
        "pozitif", "negatif", "pozitif", "negatif", "pozitif",
        "negatif", "pozitif", "negatif", "pozitif", "pozitif",
        "negatif", "pozitif", "negatif", "pozitif", "negatif",
        "pozitif", "negatif", "pozitif", "negatif", "pozitif",
        "negatif", "pozitif", "negatif", "pozitif", "negatif",
        "pozitif", "negatif", "pozitif", "negatif"
    ]
}

df = pd.DataFrame(data)

### Metin temizleme ve preprocessing işlemleri: token, pading, label encoding
# tokenizasyon
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df["text"])
word_index = tokenizer.word_index

# padding 
maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=maxlen)

print(X.shape)

# label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])


# train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

### metin temsili: word2vec
sentences = [text.split() for text in df["text"]]
word2vec_model = Word2Vec(sentences, vector_size=50, window=5, min_count=1)

embedding_dim = 50
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

### model oluşturma: rnn  train- test
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=maxlen,
                    trainable=False))
model.add(SimpleRNN(50, return_sequences = False))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

# evaluation

loss, accuracy = model.evaluate(X_test, y_test)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

def classify_sentences(sentence):

    seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq, maxlen=maxlen)

    prediction = model.predict(padded_seq)

    predicted_class = (prediction > 0.5).astype(int)
    label = "positive" if predicted_class[0][0] == 1 else "negative"

    return label

sentence = "yemekler çok tuzluydu"

result = classify_sentences(sentence)
print("Sonuç", result)

