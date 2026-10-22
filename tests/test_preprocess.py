import unittest

from irengine import STOPWORDS, preprocess, tokenize


class Preprocess(unittest.TestCase):
    def test_lowercase_and_punctuation(self):
        self.assertEqual(preprocess("Election—Commission, (EC) 2026!"), ["election", "commission", "ec", "2026"])

    def test_hyphens_split(self):
        self.assertEqual(preprocess("co-curricular self-made"), ["co", "curricular", "self", "made"])

    def test_curly_apostrophes_possessives_contractions(self):
        self.assertEqual(preprocess("The EC’s rules"), ["ec", "rules"])
        self.assertEqual(preprocess("Students’ Union"), ["students", "union"])
        self.assertEqual(preprocess("They don’t know"), ["know"])
        self.assertEqual(preprocess("I’m sure"), ["sure"])
        self.assertEqual(preprocess("‘Grey Period’"), ["grey", "period"])

    def test_curly_and_straight_apostrophes_match(self):
        self.assertEqual(preprocess("union’s don’t"), preprocess("union's don't"))

    def test_accents_and_invisible_characters(self):
        self.assertEqual(preprocess("Café"), ["cafe"])
        self.assertEqual(preprocess("word⁠joiner soft­hyphen"), ["wordjoiner", "softhyphen"])

    def test_stopwords_removed_after_normalising(self):
        self.assertIn("dont", STOPWORDS)
        self.assertIn("it", STOPWORDS)
        self.assertNotIn("well", STOPWORDS)  # "we'll" left out on purpose
        self.assertEqual(preprocess("the of and it's"), [])
        self.assertEqual(tokenize("the of"), ["the", "of"])

    def test_query_and_document_processing_identical(self):
        text = "What did the Students’ Union decide about the Grey Period?"
        self.assertEqual(preprocess(text), ["students", "union", "decide", "grey", "period"])


if __name__ == "__main__":
    unittest.main()
