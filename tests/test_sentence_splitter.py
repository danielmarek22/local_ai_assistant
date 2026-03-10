import unittest

from app.services.sentence_splitter import split_sentences


class SentenceSplitterTests(unittest.TestCase):
    def test_splits_multiple_complete_sentences(self):
        sentences, remainder = split_sentences("One. Two! Three? ")

        self.assertEqual(sentences, ["One.", "Two!", "Three?"])
        self.assertEqual(remainder, "")

    def test_returns_remainder_when_last_sentence_incomplete(self):
        sentences, remainder = split_sentences("Hello world. This is")

        self.assertEqual(sentences, ["Hello world."])
        self.assertEqual(remainder, "This is")

    def test_no_punctuation_returns_full_remainder(self):
        sentences, remainder = split_sentences("No complete sentence yet")

        self.assertEqual(sentences, [])
        self.assertEqual(remainder, "No complete sentence yet")


if __name__ == "__main__":
    unittest.main()
