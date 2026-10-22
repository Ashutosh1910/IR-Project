"""Fixed English stopword list (step 10).

This is the standard NLTK English list, written out here so the project has no
download step and the list can never change between runs. The entries are run
through the same normaliser as the text (see ``preprocess.STOPWORDS``), so
"don't" becomes "dont" and "it's" becomes "it", exactly as they would in an article.
"""

RAW_STOPWORDS = """
i me my myself we our ours ourselves you you're you've you'll you'd your yours
yourself yourselves he him his himself she she's her hers herself it it's its
itself they them their theirs themselves what which who whom this that that'll
these those am is are was were be been being have has had having do does did
doing a an the and but if or because as until while of at by for with about
against between into through during before after above below to from up down in
out on off over under again further then once here there when where why how all
any both each few more most other some such no nor not only own same so than too
very s t can will just don don't should should've now d ll m o re ve y ain aren
aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn hasn't haven
haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't
shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
""".split() + [
    # Contractions missing from the NLTK list. Forms that collapse into real
    # words once the apostrophe is removed (we'll -> "well", I'd -> "id",
    # she'll -> "shell") are deliberately left out.
    "i'm", "i've", "we've", "they're", "they've", "they'll", "they'd",
]
