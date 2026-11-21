# Reconstruction of the 16-group / 5,732-document corpus (archived 2026-11-21)

**Question.** The manuscript's real workload uses "16 of the 20 groups, 5,732
documents after dropping documents whose body has fewer than 25 tokens". The
plain filter alone applied to the public 20news-bydate-test split yields
7,275-7,294 documents over all 20 groups (depending on the token definition),
so the draft's copy was additionally filtered by dropping groups. Can that
configuration be reproduced?

**Method.** Exhaustive sweep: drop exactly 4 of the 20 groups x body-token
threshold t in [20, 120], under four tokenisations of the body (text after the
first blank line): whitespace split, \w+, [A-Za-z]+, and \w+ excluding
1-character words. Target: exactly 5,732 documents over exactly 16 groups.
Full hit lists: `reconstruction_result.txt`. Script: `../real/build_corpus.py`
(verified by SHA256 against the official sklearn checksum).

**Result.** 355 / 354 / 297 / 306 exact-count hits per tokenisation
(whitespace / \w+ / alpha / \w+ excl. 1-char). **The solutions are not
concentrated: every hit uses a distinct dropped-group combination** (355 hits =
355 different 4-group subsets under whitespace tokenisation), with thresholds
spanning 25-46 tokens. No group is common to all solutions; the most frequent
groups appear in roughly 20-25% of hits.

**Example reproduction.** Dropping {alt.atheism, comp.graphics,
comp.sys.ibm.pc.hardware, comp.sys.mac.hardware} and keeping documents with
>=35 whitespace tokens in the body yields exactly 5,732 documents over 16 groups.

**Conclusion recorded in the manuscript.** The configuration is reproducible
in count, not in membership: the counts alone do not identify the dropped-group
composition, so recorded tables are not cell-comparable with any replication,
and results on reconstructed subsets are reported separately (see the
manuscript's Data Availability statement and Limitations).

**Baseline reference.** Plain filter (body >= 25 tokens): 7,275 docs / 20 groups
(the paper's independent replication; its token definition), 7,294 docs / 20
groups (whitespace tokenisation used here).
