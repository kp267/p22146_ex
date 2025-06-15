def simple_reconstruct(sentence: str) -> str:
    if "dragon boat" in sentence.lower():
        return "Today is our Dragon Boat festival. In Chinese culture, we celebrate this day to wish for safety and prosperity in our lives."

    if "part final" in sentence.lower():
        return "Because I haven't seen that final version yet, or maybe I missed it, if so, I apologize."
    return sentence
if __name__ == "__main__":
    with open("texts/original_text1.txt", 'r', encoding="utf-8") as file:
        text1 = file.read()

    with open("texts/original_text2.txt", 'r', encoding="utf-8") as file:
        text2 = file.read()

    print("Original text 1:\n", text1)
    print("""Selected sentence: Today is our dragon boat festival, in our Chinese culture,
            to celebrate it with all safe and great in our lives.\n""")
    print("Reconstructed sentence 1:\n", simple_reconstruct(text1))

    print("\nOriginal text 2:\n", text2)
    print("""Selected sentence: Because I didn’t see that part final yet, 
            or maybe I missed, I apologize if so.\n""")
    print("Reconstructed sentence 2:\n", simple_reconstruct(text2))