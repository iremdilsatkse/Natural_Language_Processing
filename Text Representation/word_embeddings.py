# Word Embedding NLP ve makine öğrenme uygulamalarında kelimeleri sayısal vektörlere dönüştürmek için kullanılır.
# Bu vektörler, kelimelerin anlamını ve bağlamını yakalamak için kullanılır.
"""
word2vec (Google tarafından geliştirilen bir modeldir) ve GloVe (Stanford Üniversitesi tarafından geliştirilen bir modeldir) gibi popüler word embedding teknikleri vardır. Bu teknikler, kelimeleri yüksek boyutlu vektörler olarak temsil eder ve bu vektörler arasındaki benzerlikleri ölçmek için kullanılır.
fasttext (Meta tarafından geliştirilen bir modeldir) ve ELMo (Allen Institute for AI tarafından geliştirilen bir modeldir) gibi diğer popüler word embedding teknikleri de vardır. Bu teknikler, kelimelerin bağlamını ve anlamını daha iyi yakalamak için kullanılır.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess

# Örnek veri seti
sentences = [
    "Dogs are great companions",
    "Cats are independent animals",
    "Birds can fly high in the sky",
    "Fish swim in water",
    "Animals are diverse and fascinating"]

tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

# Word2Vec
word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, sg=0)

# FastText
fasttext_model = FastText(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, sg=0)

# Görselleştirme
def plot_embeddings(model, title):
    word_vectors = model.wv
    words = list(model.wv.index_to_key)[:1000]
    vectors = [word_vectors[word] for word in words]

    # PCA
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(vectors)

    # 3D Görselleştirme
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Vektörleri çiz
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2])

    # Kelimeleri etiketle
    for i, word in enumerate(words):
        ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], word, fontsize=12)

    ax.set_title(title)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.show()

plot_embeddings(word2vec_model, "Word2Vec")
plot_embeddings(fasttext_model, "FastText")