"""Cora P0.5 - a grounded, citation-gated hypothesis-card tool for the comparative biology of aging.

Everything a card says about the literature must survive a *mechanical* citation gate:
the cited PMID must exist in the local corpus, and the quote must be an exact
(whitespace-normalized) substring of that abstract. Nothing model-judged stands in
for that check in P0.5.
"""

__version__ = "0.0.5"
