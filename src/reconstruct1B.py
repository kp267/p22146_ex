from nltk.tokenize import sent_tokenize
from transformers import (T5ForConditionalGeneration, T5Tokenizer,
                          PegasusForConditionalGeneration, PegasusTokenizer,
                          BartForConditionalGeneration, BartTokenizer)

# T5 model
t5_model_name = "Vamsi/T5_Paraphrase_Paws"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# Pegasus model
pegasus_model_name = "tuner007/pegasus_paraphrase"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name)

# Bart model
bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)

# T5 paraphrasing
def paraphrase_t5(sentence, num_beams=5, num_return_sequences=1):
    input_text = f"paraphrase: {sentence} "
    encoding = t5_tokenizer.encode_plus(input_text, return_tensors="pt", padding=True, truncation=True)
    output = t5_model.generate(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        max_length=256,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        temperature=1.5,
        do_sample=True
    )
    paraphrased = t5_tokenizer.decode(output[0], skip_special_tokens=True)
    return paraphrased

# Pegasus paraphrasing
def paraphrase_pegasus(sentence):
    encoding = pegasus_tokenizer(sentence, return_tensors="pt", truncation=True, padding="longest")
    output = pegasus_model.generate(
        **encoding,
        max_length=60,
        num_beams=5,
        num_return_sequences=1,
        temperature=1.5,
        do_sample=True,
        early_stopping=True
    )
    return pegasus_tokenizer.decode(output[0], skip_special_tokens=True)

# Bart paraphrasing
def paraphrase_bart(sentence):
    inputs = bart_tokenizer([sentence], max_length=1024, return_tensors="pt",truncation=True)
    summary_ids = bart_model.generate(
        inputs["input_ids"],
        num_beams=6,
        max_length=60,
        min_length=10,
        length_penalty=1.2,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=0.9,
        do_sample=True
    )
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# applies all three pipelines
def reconstruct_text(text, paraphrasing_function):
    sentences = sent_tokenize(text)
    reconstructed = []
    for sent in sentences:
        try:
            reconstructed.append(paraphrasing_function(sent))
        except Exception:
            reconstructed.append(sent)
    return " ".join(reconstructed)

if __name__ == "__main__":
    with open("texts/original_text1.txt", "r", encoding="utf-8") as file:
        text1 = file.read()

    with open("texts/original_text2.txt", "r", encoding="utf-8") as file:
        text2 = file.read()

    models = {
        "T5": paraphrase_t5,
        "Pegasus": paraphrase_pegasus,
        "Bart": paraphrase_bart
    }

    for i, original_text in enumerate([text1, text2], start=1):
        print(f"\nReconstructed Versions for Text {i}: ")
        for model_name, func in models.items():
            reconstructed = reconstruct_text(original_text, func)
            print(f"\n- {model_name}:\n{reconstructed}\n")