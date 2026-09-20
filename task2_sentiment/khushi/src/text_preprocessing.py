"""Task 2 (Step 2): text preprocessing for Khushi's Yelp Polarity pipeline.

Pipeline order, applied to raw review text:
  1. lowercase
  2. expand negation contractions ("can't" -> "can not", "didn't" -> "did not", ...)
     so negation is captured as its own word before punctuation is stripped,
     instead of relying on apostrophe-stripped forms surviving by accident.
  3. normalize whitespace
  4. remove punctuation / special characters (keep only a-z, 0-9, whitespace)
  5. tokenize (whitespace split on the already-cleaned text)
  6. remove standard English stopwords, except a broader negation exception
     list (not just "not"/"no"/"nor") that stays in, since negation carries
     sentiment signal.

No stemming or lemmatization in this initial pipeline. No pretrained
tokenizers are used anywhere here.
"""

import re

from nltk.corpus import stopwords

# Negation contractions expanded to two words *before* punctuation removal,
# so negation is never accidentally lost to apostrophe stripping.
NEGATION_CONTRACTIONS = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "shan't": "shall not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "mustn't": "must not",
    "mightn't": "might not",
    "needn't": "need not",
    "ain't": "is not",
}
_CONTRACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in NEGATION_CONTRACTIONS) + r")\b"
)

# Broader standard negation set (per Khushi's choice): these are exempted from
# stopword removal because they carry sentiment-critical negation meaning.
NEGATION_EXCEPTIONS = {
    "no", "not", "nor", "never", "none", "nobody", "nothing",
    "neither", "nowhere", "cannot", "without",
}

_ENGLISH_STOPWORDS = set(stopwords.words("english"))
STOPWORDS = _ENGLISH_STOPWORDS - NEGATION_EXCEPTIONS

_WHITESPACE_PATTERN = re.compile(r"\s+")
_NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9\s]")

# This dataset stores literal escape sequences as two literal characters
# (backslash + letter) rather than actual control characters. Left alone,
# punctuation removal strips the backslash and leaves a stray letter behind
# (e.g. literal "\n" -> "n" token). Narrowly unescape just these three
# before anything else, rather than using a general-purpose escape decoder.
_ESCAPE_ARTIFACT_PATTERN = re.compile(r"\\n|\\r|\\t")


def expand_negation_contractions(text):
    return _CONTRACTION_PATTERN.sub(lambda m: NEGATION_CONTRACTIONS[m.group(0)], text)


def clean_text(text):
    text = _ESCAPE_ARTIFACT_PATTERN.sub(" ", text)
    text = text.lower()
    text = expand_negation_contractions(text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    text = _NON_ALPHANUMERIC_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def tokenize(cleaned_text):
    return cleaned_text.split()


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]


def preprocess(text):
    """Raw review text -> list of tokens, after the full Step 2 pipeline."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    return tokens
