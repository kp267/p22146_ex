from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from reconstruct1B import (paraphrase_t5,
                           paraphrase_pegasus,
                           paraphrase_bart,
                           reconstruct_text)
import numpy as np
import re
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import gensim.downloader as api

glove_model = api.load("glove-wiki-gigaword-100")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# visualises the embeddings with PCA
def visualise_embeddings(texts, labels):
    embeddings = embedding_model.encode(texts)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    plt.figure(figsize=(8,6))
    for i, label in enumerate(labels):
        plt.scatter(reduced[i,0], reduced[i,1], label=label)
    plt.legend()
    plt.title("Semantic Embedding Visualisation (PCA)")
    plt.show()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in stopwords.words("english")]
    return " ".join(tokens)

def word_embedding(text, model):
    tokens = preprocess(text).split()
    valid_tokens = [token for token in tokens if token in model]
    if not valid_tokens:
        return np.zeros(model.vector_size)
    return np.mean([model[token] for token in valid_tokens], axis=0)

def compute_similarity_word_emb(text1, text2, model):
    vec1 = word_embedding(text1, model)
    vec2 = word_embedding(text2, model)
    return  cosine_similarity([vec1], [vec2])[0][0]

# visualisation with t-SNE
def visualise_tsne(texts, labels):
    embeddings = embedding_model.encode(texts)
    tsne = TSNE(n_components=2, perplexity=3, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8,6))
    for i, label in enumerate(labels):
        plt.scatter(reduced[i,0], reduced[i,1], label=label)
    plt.legend()
    plt.title("t-SNE Semantic Embedding Visualisation")
    plt.show()

def show_top_words(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    feature_names = vectorizer.get_feature_names_out()
    diff = tfidf.toarray()[1] - tfidf.toarray()[0]
    top_indices = diff.argsort()[-10:]
    for i in reversed(top_indices):
        print(feature_names[i], "-", round(diff[i], 3))

def text_to_vec(text, model):
    words = text.lower().split()
    word_vecs = [model[w] for w in words if w in model]
    if len(word_vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vecs, axis=0)

def compute_glove_similarity(text1, text2):
    vec1 = text_to_vec(text1, glove_model)
    vec2 = text_to_vec(text2, glove_model)
    return cosine_similarity([vec1], [vec2])[0][0]

if __name__ == "__main__":
    with open("texts/original_text1.txt", "r", encoding="utf-8") as file:
        text1 = file.read()

    with open("texts/original_text2.txt", "r", encoding="utf-8") as file:
        text2 = file.read()

    # reconstructing both texts with each model separately
    text1_t5 = reconstruct_text(text1, paraphrase_t5)
    text1_pegasus  = reconstruct_text(text1, paraphrase_pegasus)
    text1_bart = reconstruct_text(text1, paraphrase_bart)

    text2_t5 = reconstruct_text(text2, paraphrase_t5)
    text2_pegasus = reconstruct_text(text2, paraphrase_pegasus)
    text2_bart = reconstruct_text(text2, paraphrase_bart)

    # visualising results for both texts
    visualise_embeddings(
        [text1, text1_t5, text1_pegasus, text1_bart],
        ["Original Text 1", "T5 Reconstruction", "Pegasus Reconstruction", "Bart Reconstruction"]
    )

    visualise_embeddings(
        [text2, text2_t5, text2_pegasus, text2_bart],
        ["Original Text 2", "T5 Reconstruction", "Pegasus Reconstruction", "Bart Reconstruction"]
    )

    print("\nWord Embedding Similarity (GloVe)")
    print("Text 1 vs T5: ", compute_similarity_word_emb(text1, text1_t5, glove_model))
    print("Text 1 vs Pegasus: ", compute_similarity_word_emb(text1, text1_pegasus, glove_model))
    print("Text 1 vs Bart: ", compute_similarity_word_emb(text1, text1_bart, glove_model))

    print("Text 2 vs T5: ", compute_similarity_word_emb(text2, text2_t5, glove_model))
    print("Text 2 vs Pegasus: ", compute_similarity_word_emb(text2, text2_pegasus, glove_model))
    print("Text 2 vs Bart: ", compute_similarity_word_emb(text2, text2_bart, glove_model))

    print("\nTop changed words for Text 1 with T5: ")
    show_top_words(text1, text1_t5)
    print("\nTop changed words for Text 1 with Pegasus: ")
    show_top_words(text1, text1_pegasus)
    print("\nTop changed words for Text 1 with Bart: ")
    show_top_words(text1, text1_bart)

    print("\nTop changed words for Text 2 with T5: ")
    show_top_words(text2, text2_t5)
    print("\nTop changed words for Text 2 with Pegasus: ")
    show_top_words(text2, text2_pegasus)
    print("\nTop changed words for Text 2 with Bart: ")
    show_top_words(text2, text2_bart)

    # t-SNE visualisation
    visualise_tsne(
        [text1, text1_t5, text1_pegasus, text1_bart],
        ["Original Text 1", "T5", "Pegasus", "Bart"]
    )

    visualise_tsne(
        [text2, text2_t5, text2_pegasus, text2_bart],
        ["Original Text 2", "T5", "Pegasus", "Bart"]
    )