from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from reconstruct1A import simple_reconstruct
from reconstruct1B import (reconstruct_text,
                           paraphrase_t5,
                           paraphrase_pegasus,
                           paraphrase_bart)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_similarity(text1, text2):
    emb1 = embedding_model.encode([text1])
    emb2 = embedding_model.encode([text2])
    return cosine_similarity(emb1, emb2)[0][0]

if __name__ == "__main__":
    with open("texts/original_text1.txt", "r", encoding="utf-8") as file:
        text1 = file.read()

    with open("texts/original_text2.txt", "r", encoding="utf-8") as file:
        text2 = file.read()

    text1_simple_reconstruct = simple_reconstruct(text1)
    text1_t5 = reconstruct_text(text1, paraphrase_t5)
    text1_pegasus = reconstruct_text(text1, paraphrase_pegasus)
    text1_bart = reconstruct_text(text1, paraphrase_bart)

    text2_simple_reconstruct = simple_reconstruct(text2)
    text2_t5 = reconstruct_text(text2, paraphrase_t5)
    text2_pegasus = reconstruct_text(text2, paraphrase_pegasus)
    text2_bart = reconstruct_text(text2, paraphrase_bart)

    print("Similarity comparison for Text 1:")
    print("Reconstruct 1A: ", compute_similarity(text1, text1_simple_reconstruct))
    print("Reconstruct T5: ", compute_similarity(text1, text1_t5))
    print("Reconstruct Pegasus: ", compute_similarity(text1, text1_pegasus))
    print("Reconstruct Bart: ", compute_similarity(text1, text1_bart))

    print("Similarity comparison for Text 2:")
    print("Reconstruct 1A: ", compute_similarity(text2, text2_simple_reconstruct))
    print("Reconstruct T5: ", compute_similarity(text2, text2_t5))
    print("Reconstruct Pegasus: ", compute_similarity(text2, text2_pegasus))
    print("Reconstruct Bart: ", compute_similarity(text2, text2_bart))