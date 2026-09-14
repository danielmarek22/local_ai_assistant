import re

SENTENCE_END = re.compile(r"([.!?])\s+")


def split_sentences(buffer: str):
    parts = SENTENCE_END.split(buffer)
    sentences = []

    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i] + parts[i + 1]
        sentences.append(sentence.strip())

    remainder = parts[-1] if len(parts) % 2 == 1 else ""
    return sentences, remainder
