"""
Evidence retrieval — filtered-first, then similarity. NOT LLM: TF-IDF, no
model call.

Docs are pre-filtered by region mention + recency BEFORE ranking by
similarity. This filtering is what makes the evidence defensible instead of
"vaguely related documents from anywhere, anytime."

TWO FIXES made after review, both worth being explicit about:

1. VOCABULARY MISMATCH BUG: a caller query like "stockout" shares zero
   word-level tokens with a document that says "out of stock" or "restock"
   -- literally 0.0 cosine similarity, despite being the most relevant
   document available. We fix this by expanding the query with the SAME
   cause-vocabulary already defined in engine/confidence.py's
   KEYWORD_TO_CAUSE, rather than inventing a second, separate synonym list
   or switching to a fuzzier character-n-gram matcher (which we tested and
   rejected -- it also inflated noise-document scores on this dataset).
   One vocabulary, used consistently by both retrieval and ranking.

2. SCORE PRESENTATION: raw TF-IDF cosine similarity on a handful of short
   documents is naturally small (0.05-0.15 range) even for a genuinely
   strong match, which reads as "weak evidence" to anyone looking at the
   raw number in isolation. We report a RELATIVE relevance (this
   document's score divided by the best score among the already-filtered
   candidates) alongside the raw score, so "this is the strongest match we
   found after filtering by time+region" is legible without requiring the
   reader to know what a "good" cosine similarity looks like. A floor
   still excludes documents that don't meaningfully match anything.
"""
import os
import re
import glob
from datetime import datetime, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from engine import DATA_DIR
from engine.confidence import KEYWORD_TO_CAUSE

TODAY = datetime(2026, 8, 25)  # kept fixed so the demo dataset stays reproducible
MIN_ABSOLUTE_RELEVANCE = 0.02   # below this, a document is excluded outright, not just ranked low


def retrieve_evidence(region: str, window_days: int = 21, query: str = ""):
    doc_paths = glob.glob(os.path.join(DATA_DIR, "docs", "*.txt"))
    candidates = []
    cutoff = TODAY - timedelta(days=window_days)

    for path in doc_paths:
        text = open(path).read()
        date_match = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", text)
        region_match = re.search(r"Region:\s*(\w+)", text)
        doc_date = datetime.strptime(date_match.group(1), "%Y-%m-%d") if date_match else None
        doc_region = region_match.group(1) if region_match else None

        if doc_date and doc_date >= cutoff and doc_region in (region, "National"):
            candidates.append({"file": os.path.basename(path), "text": text, "date": doc_date})

    if not candidates:
        return []

    # Fix #1: expand the query with the SAME vocabulary confidence.py uses
    # for cause keyword matching, so a query for "stockout" also catches a
    # document that only ever says "out of stock" or "restock."
    expanded_query = query + " " + " ".join(KEYWORD_TO_CAUSE.keys())

    corpus = [c["text"] for c in candidates] + [expanded_query]
    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform(corpus)
    sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()

    for c, s in zip(candidates, sims):
        c["relevance_score"] = round(float(s), 3)  # raw cosine similarity, kept for audit/transparency

    candidates = [c for c in candidates if c["relevance_score"] >= MIN_ABSOLUTE_RELEVANCE]
    if not candidates:
        return []

    # Fix #2: relative relevance + a qualitative label, computed only among
    # documents that already cleared the absolute floor above -- this isn't
    # inflating weak evidence, it's making "strongest match in an
    # already-filtered set" legible instead of an unexplained small number.
    max_score = max(c["relevance_score"] for c in candidates)
    for c in candidates:
        rel = c["relevance_score"] / max_score if max_score else 0.0
        c["relative_relevance"] = round(rel, 3)
        c["relevance_label"] = "High" if rel >= 0.7 else "Medium" if rel >= 0.4 else "Low"

    candidates.sort(key=lambda c: -c["relative_relevance"])
    return candidates[:5]
