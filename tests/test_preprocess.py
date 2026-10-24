import unittest

from irengine import STOPWORDS, preprocess, tokenize
from irengine.porter import stem


def unstemmed(text):
    """Steps 1-5 only, so the normalisation tests do not depend on the stemmer."""
    return [tok for tok in tokenize(text) if tok not in STOPWORDS]


class Preprocess(unittest.TestCase):
    def test_lowercase_and_punctuation(self):
        self.assertEqual(unstemmed("Election—Commission, (EC) 2026!"), ["election", "commission", "ec", "2026"])

    def test_hyphens_split(self):
        self.assertEqual(unstemmed("co-curricular self-made"), ["co", "curricular", "self", "made"])

    def test_curly_apostrophes_possessives_contractions(self):
        self.assertEqual(unstemmed("The EC’s rules"), ["ec", "rules"])
        self.assertEqual(unstemmed("Students’ Union"), ["students", "union"])
        self.assertEqual(unstemmed("They don’t know"), ["know"])
        self.assertEqual(unstemmed("I’m sure"), ["sure"])
        self.assertEqual(unstemmed("‘Grey Period’"), ["grey", "period"])

    def test_curly_and_straight_apostrophes_match(self):
        self.assertEqual(preprocess("union’s don’t"), preprocess("union's don't"))

    def test_accents_and_invisible_characters(self):
        self.assertEqual(unstemmed("Café"), ["cafe"])
        self.assertEqual(unstemmed("word⁠joiner soft­hyphen"), ["wordjoiner", "softhyphen"])

    def test_stopwords_removed_after_normalising(self):
        self.assertIn("dont", STOPWORDS)
        self.assertIn("it", STOPWORDS)
        self.assertNotIn("well", STOPWORDS)  # "we'll" left out on purpose
        self.assertEqual(preprocess("the of and it's"), [])
        self.assertEqual(tokenize("the of"), ["the", "of"])

    def test_stopwords_removed_before_stemming(self):
        # "this" would stem to "thi", which is not on the list
        self.assertEqual(preprocess("this was"), [])

    def test_full_pipeline_stems(self):
        self.assertEqual(preprocess("Election—Commission, (EC) 2026!"), ["elect", "commiss", "ec", "2026"])
        self.assertEqual(preprocess("The EC’s rules"), ["ec", "rule"])
        self.assertEqual(preprocess("HuEl HuEls"), ["huel", "huel"])

    def test_query_and_document_processing_identical(self):
        text = "What did the Students’ Union decide about the Grey Period?"
        self.assertEqual(preprocess(text), ["student", "union", "decid", "grei", "period"])


class PorterStemmer(unittest.TestCase):
    """Examples given for each step in Porter (1980)."""

    def check(self, pairs):
        for word, expected in pairs:
            with self.subTest(word=word):
                self.assertEqual(stem(word), expected)

    def test_step1(self):
        self.check([
            ("caresses", "caress"), ("ponies", "poni"), ("ties", "ti"), ("caress", "caress"),
            ("cats", "cat"), ("feed", "feed"), ("agreed", "agre"), ("plastered", "plaster"),
            ("bled", "bled"), ("motoring", "motor"), ("sing", "sing"), ("conflated", "conflat"),
            ("troubled", "troubl"), ("sized", "size"), ("hopping", "hop"), ("tanned", "tan"),
            ("falling", "fall"), ("hissing", "hiss"), ("fizzed", "fizz"), ("failing", "fail"),
            ("filing", "file"), ("happy", "happi"), ("sky", "sky"),
        ])

    def test_step2(self):
        self.check([
            ("relational", "relat"), ("conditional", "condit"), ("rational", "ration"),
            ("valenci", "valenc"), ("hesitanci", "hesit"), ("digitizer", "digit"),
            ("conformabli", "conform"), ("radicalli", "radic"), ("differentli", "differ"),
            ("vileli", "vile"), ("analogousli", "analog"), ("vietnamization", "vietnam"),
            ("predication", "predic"), ("operator", "oper"), ("feudalism", "feudal"),
            ("decisiveness", "decis"), ("hopefulness", "hope"), ("callousness", "callous"),
            ("formaliti", "formal"), ("sensitiviti", "sensit"), ("sensibiliti", "sensibl"),
        ])

    def test_step3_and_4(self):
        self.check([
            ("triplicate", "triplic"), ("formative", "form"), ("formalize", "formal"),
            ("electriciti", "electr"), ("electrical", "electr"), ("hopeful", "hope"),
            ("goodness", "good"), ("revival", "reviv"), ("allowance", "allow"),
            ("inference", "infer"), ("airliner", "airlin"), ("gyroscopic", "gyroscop"),
            ("adjustable", "adjust"), ("defensible", "defens"), ("irritant", "irrit"),
            ("replacement", "replac"), ("adjustment", "adjust"), ("dependent", "depend"),
            ("adoption", "adopt"), ("communism", "commun"), ("activate", "activ"),
            ("angulariti", "angular"), ("homologous", "homolog"), ("effective", "effect"),
            ("bowdlerize", "bowdler"),
        ])

    def test_step5(self):
        self.check([("probate", "probat"), ("rate", "rate"), ("cease", "ceas"),
                    ("controll", "control"), ("roll", "roll")])

    def test_longest_suffix_only(self):
        # "agreement": -ement leaves m = 1, so nothing is removed (-ment is not tried)
        self.assertEqual(stem("agreement"), "agreement")

    def test_short_words_unchanged(self):
        self.check([("us", "us"), ("is", "is"), ("s", "s"), ("a", "a")])


if __name__ == "__main__":
    unittest.main()
