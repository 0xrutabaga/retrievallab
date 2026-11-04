"""Tokenisation shared by the lexical retrievers.

Deliberately simple and dependency-free: lowercase, split on anything that is not an ASCII
letter or digit, drop a short English stopword list. No stemming, so "token" and "tokens" are
different terms; this is a known weakness recorded in the README limitations.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

# Function words plus the RFC 2119 keywords (MUST, SHOULD, MAY, ...), which occur in most
# normative sentences and so carry almost no ranking signal.
_STOPWORD_TEXT = """
a an and are as at be but by can do does for from has have how if in into is it its may
must not of on or should shall such than that the their then there these this those to
was what when where which who why will with would
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
