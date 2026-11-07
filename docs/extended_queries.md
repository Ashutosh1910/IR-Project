# Extended query set (Q13–Q40)

The 28 new queries in `queries_extended.json` extend the 12 frozen queries in
`queries.json` for a second run of the main evaluation and the attack experiments.
The frozen labels for Q1–Q12 (`qrels_final.csv`, checksum in `labels_frozen.json`)
are not touched.

## The queries

| Group | Meant to have relevant articles | Meant to have none |
|---|---|---|
| small (1–2 words) | Q13–Q18 | Q33–Q35 |
| medium (3–5 words) | Q19–Q25 | Q36–Q38 |
| large (questions) | Q26–Q32 | Q39–Q40 |

The queries meant to have no relevant articles use words that do appear in the
collection, so they still retrieve articles. Those words appear only in passing or
in a different sense; for example, Porter stems "ragging" (Q33) to "rag", which
matches the "Audi Rag" election-debate articles. Whether a query really has no
relevant article is decided by its labels, not by this plan.

## Labelling run

- Prompt and guideline: the same as Q1–Q12 (`docs/labeling_prompt.md`,
  `docs/relevance_guideline.md`).
- Model and exact version: Claude Opus 5.5 (`claude-opus-5-5`).
- How it was run: as for Q1–Q12, in a Claude Code session with no API key. Each article
  was read once and judged for all 28 new queries together: 100 passes, not 2,800
  separate calls. Pairs with no connection to an article were written with label 0,
  excerpt `none`, and a reason naming the article's topic.
- Date of run: 2026-11-07.
- What the session could see: the article texts, the queries and descriptions, and
  word counts from a text search done while drafting the queries (how many articles
  contain a word). No retrieval scores or rankings were computed or seen before
  labelling.
- Output: `results/llm_labels_extended.jsonl` (2,800 records).
- Checks: `python check_labels.py --labels results/llm_labels_extended.jsonl
  --queries queries_extended.json --manual manual_labels_extended.csv
  --qrels qrels_extended.csv --csv results/llm_labels_extended.csv`: all 2,800 pairs
  present, every excerpt found word for word, 16 pairs flagged uncertain.

## LLM result before review

- 45 relevant pairs across Q13–Q32. Q27 (COVID vaccination drive) had none: its two
  closest articles were flagged uncertain.
- No relevant pair on Q33–Q40. Two of these queries had uncertain pairs (Q36 1, Q38 2).

## Hand review and freeze

- All 16 uncertain pairs were reviewed by one reviewer (Ashu) on 2026-11-07, in a review
  page like the one used for Q1–Q12. Decisions are in `manual_labels_extended.csv`
  (stage `uncertain`).
- 4 of the 16 labels changed, all from not relevant to relevant:

  | Query | Document | LLM | Manual |
  |---|---:|---:|---:|
  | Q14 | 7491 | 0 | 1 |
  | Q27 | 5304 | 0 | 1 |
  | Q38 | 5305 | 0 | 1 |
  | Q38 | 7269 | 0 | 1 |

- Result: 49 relevant pairs. Q27 now has one relevant article. **Q38 (hackathon
  results) now has two, so it is treated as an ordinary query, and 7 queries have no
  relevant article: Q33–Q37, Q39 and Q40.**
- Frozen on 2026-11-07: `qrels_extended.csv`, with its SHA-256 in
  `labels_frozen_extended.json`. Both runners check it, and the Q1–Q12 checksum, before
  running.

## Runs

```
python run_epc.py --queries extended        # writes results/epc_extended/
python run_attacks.py --queries extended    # writes results/attacks_extended/
```

Without `--queries extended` both scripts run Q1–Q12 exactly as before; a rerun
reproduces every saved Q1–Q12 output. Target selection draws from one seeded
generator over the queries in file order, so the 72 Q1–Q12 targets are the same in
both runs and the extended run adds 168.

## Main evaluation, Q1–Q40

Averages are over the 33 queries with a relevant article; the 7 without one are left
out (step 17).

| System | P@10 | R@10 | F1@10 | MAP |
|---|---:|---:|---:|---:|
| cosine | 0.2333 | 0.9015 | 0.3561 | 0.7835 |
| pivoted | 0.2303 | 0.8939 | 0.3518 | 0.7872 |

Pivoted has higher AP on 5 queries, lower on 5 and equal on 23 (randomization test
p = 0.47 for AP, 1.00 for P@10). On the 7 empty queries the two methods put almost the
same articles in the top 10 (overlap 10/10 on five, 9/10 on Q34; Q33 retrieves only
3 articles).

## Attacks, Q1–Q40

240 targets (6 per query), 94 of them not retrieved before the attack; 11,520 trials.

- **Repetition** works almost always under both methods: ×5 puts the target in the top
  10 in 97.9% (cosine) and 98.8% (pivoted) of trials, ×10 and above in 100%. Neither
  method moves targets up more often (sign test p ≥ 0.74).
- **Padding** never puts a target in the top 10 (the built-in check confirms it never
  raises a score). Pivoted loses fewer places: at 200% padding the target fell less
  under pivoted for 92 targets and less under cosine for 17 (p < 0.0001). Score kept at
  200%: 0.575 (cosine) vs 0.629 (pivoted).
- **Combined**: pivoted is equal or higher in every cell of the 4 × 4 grid. Over the 16
  cells, 22 targets entered the top 10 in more cells under pivoted and none under
  cosine (p = 4.8e-07).
- **Queries with no relevant article** (42 targets): every repetition and combined
  attack succeeded under both methods (100%). On queries with a relevant article
  (198 targets) ×5 repetition succeeded in 97.5% / 98.5%. All 8 failed ×5 trials were on
  Q1, Q4 and Q5 and ended at ranks 11–14.
- **ΔMAP** per edited article is smaller than in the Q1–Q12 run (about −0.004 to −0.008
  under repetition) because MAP is now averaged over 33 queries.

Figures (`python plot_attacks.py --results results/attacks_extended --name attack_extended`),
in `docs/figures/`, as PNG and PDF, on the same scales as the Q1–Q12 figures:

| Figure | Shows |
|---|---|
| `attack_extended_repetition` | success % vs repetitions, all targets and long articles |
| `attack_extended_padding` | score kept and median rank change vs padding % |
| `attack_extended_combined` | success % over the 4 × 4 grid per method, and the difference |
| `attack_extended_quality` | mean ΔMAP per level (33 queries) |
| `attack_extended_query_type` | queries with a relevant article vs queries with none |

## Compared with the Q1–Q12 run

No attack finding changes direction. The main evaluation's small advantage for pivoted
mostly disappears.

| | Q1–Q12 | Q1–Q40 |
|---|---|---|
| MAP cosine / pivoted | 0.7109 / 0.7289 | 0.7835 / 0.7872 |
| AP better / worse / equal (pivoted) | 4 / 2 / 6, p = 0.13 | 5 / 5 / 23, p = 0.47 |
| P@10 cosine / pivoted | 0.2667 / 0.2667 | 0.2333 / 0.2303 |
| Relevant articles that are long | 19 of 43 (44%) | 35 of 92 (38%) |
| Repetition ×5 success, cosine / pivoted | 93.1 / 95.8% | 97.9 / 98.8% |
| Padding 200%: lost fewer places under pivoted / cosine | 26 / 4, p = 0.0001 | 92 / 17, p < 0.0001 |
| Combined: more cells under pivoted / cosine | 15 / 0, p = 6.1e-05 | 22 / 0, p = 4.8e-07 |

- **Main evaluation.** Pivoted's MAP lead shrinks from +0.018 to +0.004 and stays far from
  significant. P@10, R@10 and F1@10, tied on Q1–Q12, now favour cosine very slightly; the
  whole difference is one query, Q20, where pivoted moves a short relevant article
  (8370, 1,293 words) from rank 9 to 11 and lets in a 4,343-word non-relevant HuEl guide.
  Relevant articles across Q1–Q40 are less skewed towards long ones (mean 1,767 words,
  against 1,930 for Q1–Q12 and 1,709 for the collection), so pivoting has less to gain. Pivoted still gives
  more top-10 slots to long articles (129 vs 115) and fewer to short ones (74 vs 88).
- **Repetition.** Success rises for both methods and the gap at ×5 narrows from 2.8 to
  0.8 points; per target, neither method moves articles up more often in either run.
- **Padding and combined.** The same direction with much smaller p-values, because there
  are more targets.
- **ΔMAP** per edited article is smaller because MAP is averaged over 33 queries, not 12.
