# Eval log

Two kinds of run live here:

1. **Retrieval rankings** — the Phase 4/5 scorer experiment (keyword vs
   embedding vs hybrid, see ROADMAP). One entry per JD: the brief, what
   *should* surface, and what each scorer actually did. Five minutes per
   JD; the point is that "scorer X is better" becomes a measured recall@k
   claim rather than an impression from one run.
2. **Extractor runs** — one row per `extract_eval` run, scored against the
   hand-written brief for the case.
3. **Ranker runs** — `rank_eval` against hand labels on a cached pool of
   postings: precision@5 of the shortlist and stage 1's recall.

Format per retrieval case: expected atoms (unordered, the ones a human would
put on the CV for this job), then one ranking block per scorer version.

## Reading this log

Two changes on 2026-09-20 split the log in half. Rows either side of them are
not directly comparable:

* **Brief alignment.** The hand-written briefs (`data/*.brief.yaml`) were
  rewritten to follow the same rules the Extractor's prompt states, so the
  eval compares like with like. See "Brief alignment" below.
* **Repeats.** `extract_eval` extracts each case three times (`--runs`,
  default 3) and reports mean (min-max) plus, per expected item, the number
  of runs it survived. Single-run rows (E1-E7) are one sample each and should
  be read as such.
* **Eval matching.** `_key()` now drops a trailing parenthesised acronym when
  it spells the initials of the words before it, so "retrieval-augmented
  generation (RAG)" and "retrieval-augmented generation" score as one item
  instead of a miss plus an extra. A bracketed word that is not the initials
  is untouched.

Three later changes matter when comparing multi-run rows:

* **Brief revisions.** Both briefs were corrected after the multi-run evals
  exposed items the alignment had missed (see "Revisions after the
  multi-run evals"). The `brief` column in the multi-run table names the
  revision each row was scored against.
* **`repair()` (prompt v3.1).** Three rule-4 violations are now fixed in code
  before `verify()` runs: bare generic words, droppable tails, "X" next to
  "X models". Precision from v3.1 on is partly code, by design. Compliance is
  still measured: the report's "repaired:" counts are how often the model
  broke each rule.
* **Attempts and retry reasons** are reported from v3 on. Earlier rows
  cannot say why a run cost more tokens.

One change on 2026-09-23 affects everything that tokenises:

* **Keyword v6.** `tokenize()` drops hyphens inside words ("e-commerce" ==
  "eCommerce" == "ecommerce", except digit ranges like "3-5") and has an
  alias table for one-concept slash terms ("A/B testing" was
  `['b', 'testing']`). Every consumer of the tokenizer changed with it:
  retrieval, the Tailor's skill check, the Extractor's `repair()`/`verify()`,
  and the eval's `_key()`, which now also ignores spacing. Retrieval blocks
  labelled v5 and v6 differ only where a hyphen or slash spelling differs
  between a brief and an atom. Extractor rows are bridged by "Keyword v6 —
  M7, M8 re-scored", which scores the same saved briefs under both.

---

## 001 — eBay, Data Scientist, Product Knowledge (`data/sample_jd.brief.yaml`)

**Expected in top 5:** `paypal-fra-insight-store` (knowledge graph, LLMs),
`target-<fp-growth affinity model>` (e-commerce catalog, recommendations),
`self-text2sql-memory-agent` (SQL, LLMs, agents), `target-goa-dmo-mismatch`
(e-commerce, SQL at scale), plus one classification atom.

**Keyword v1** — 13 must-haves joined into one `search()` query, no plural
normalisation, pre-tagging (2026-09):

```
1.  22.0  target-cross-functional-communication
2.  18.0  paypal-fra-insight-store
3.  15.0  paypal-rule-governance-analytics      <- false positive: "catalog" (rules, not products)
4.  13.0  self-attrition-prediction
5.  12.0  target-goa-dmo-mismatch
6.   8.0  pub-cyberbully-1dcnn-lstm
7.   6.5  self-disasterbert-tweet-classification
8.   6.0  self-text2sql-memory-agent           <- too low: tagged text-to-sql, not SQL
--    --  target FP-Growth atom                <- MISSING: no e-commerce / recommendations tags
```
Recall@5 = 2/5. Diagnosed: generic-token flattening (joined query), no
plural handling (`LLMs`/`LLM`, `models`/`model`), missing tags.

**Keyword v2** — per-must-have `retrieve_for_brief`, plural
normalisation, after tagging:

```
1.  34.5  target-cross-functional-communication   <- overclaims: "prompt engineering" via token "engineering"
2.  29.0  paypal-fra-insight-store
3.  20.5  target-fp-growth-affinity-model
4.  17.5  self-attrition-prediction
5.  16.5  target-goa-dmo-mismatch
6.  15.0  paypal-rule-governance-analytics
7.  13.5  self-text2sql-memory-agent
8.  11.5  pub-cyberbully-1dcnn-lstm
```
Recall@5 = 4/5. Diagnosed: `covers` counted any token overlap as
coverage; generic tokens still flattening via partial matches.

**Keyword v3** — strict coverage (all tokens), partial matches scaled by
fraction; brief still using bundled phrases:

```
1.  27.0  paypal-fra-insight-store               covers: Python, knowledge graph, LLMs
2.  24.2  target-cross-functional-communication  covers: data analytics, cross-functional collaboration
3.  12.5  self-text2sql-memory-agent             covers: SQL, Python, LLMs
4.  12.4  self-attrition-prediction              covers: Python, classification models
5.  11.8  target-goa-dmo-mismatch                covers: SQL, data analytics
6.  11.4  target-fp-growth-affinity-model        covers: -      <- bundled must-haves can't be fully matched
7.  10.0  paypal-rule-governance-analytics       covers: SQL, Python
8.   9.1  pub-cyberbully-1dcnn-lstm              covers: classification models
```
Diagnosed: strictness exposed bundled requirements in the *brief*
("e-commerce search and recommendations"). Fix is in the brief, not the scorer.

**Keyword v4** — one concept per must-have:

```
1.  27.0  paypal-fra-insight-store               covers: Python, knowledge graph, LLMs
2.  23.5  target-cross-functional-communication  covers: data analytics, cross-functional collaboration
3.  16.2  target-fp-growth-affinity-model        covers: e-commerce, recommendations, product catalog
4.  14.8  target-goa-dmo-mismatch                covers: SQL, data analytics, e-commerce
5.  12.5  self-text2sql-memory-agent             covers: SQL, Python, LLMs
6.  10.5  paypal-rule-governance-analytics       covers: SQL, Python
7.   9.5  self-attrition-prediction              covers: Python, classification
8.   9.5  pub-cyberbully-1dcnn-lstm              covers: classification
```
Recall@5 = 4/5. Remaining oddity: a soft-skill atom at #2 because soft
requirements competed with technical ones in retrieval.

**Keyword v5** — soft skills routed to `soft_skills[]` (retrieval ignores
them). **Phase 3 baseline. Pre-alignment brief (13 must-haves, later 17).**

```
1.  27.0  paypal-fra-insight-store               covers: Python, knowledge graph, LLMs
2.  16.2  target-fp-growth-affinity-model        covers: e-commerce, recommendations, product catalog
3.  14.8  target-goa-dmo-mismatch                covers: SQL, data analytics, e-commerce
4.  12.5  self-text2sql-memory-agent             covers: SQL, Python, LLMs
5.  12.0  target-cross-functional-communication  covers: data analytics
6.  10.5  paypal-rule-governance-analytics       covers: SQL, Python
7.   9.5  pub-cyberbully-1dcnn-lstm              covers: classification
8.   9.0  self-attrition-prediction              covers: Python, classification
```
Recall@5 = 4/5 (attrition at #8). Uncovered must-haves: taxonomy,
ontology, prompt engineering, model evaluation — true gaps, the Judge
should report them. Known false positive carried forward:
`rule-governance-analytics` on nothing specific (SQL/Python only).

Iteration summary: six fixes, three in data (tags, brief phrasing, brief
buckets), three in query construction (per-requirement, plurals, strict
coverage), zero in the `search()` scorer. The embedding/hybrid experiment
should be measured against this baseline, not v1.

**Keyword v5, aligned brief** (same scorer; brief now 26 must-haves in the
JD's own words — 2026-09-20):

```
1.  30.0  paypal-fra-insight-store               covers: Python, automation, knowledge graph, LLMs
                                                 supports: retrieval-augmented generation (RAG)
2.  15.0  target-cross-functional-communication  covers: data analytics
3.  13.5  target-goa-dmo-mismatch                covers: SQL, data analytics
4.  12.5  self-text2sql-memory-agent             covers: SQL, Python, LLMs
5.  12.0  paypal-rule-governance-analytics       covers: SQL, Python, catalog
6.  11.5  pub-cyberbully-1dcnn-lstm              covers: classification, embeddings
7.   9.5  target-fp-growth-affinity-model        covers: recommendations, catalog
8.   9.0  self-attrition-prediction              covers: Python, classification
```
Recall@5 = 3/5, down from 4/5 — **a known, temporary cost of the alignment,
not a scorer regression.** The brief now says `eCommerce` (the JD's spelling),
which no atom matches until `tokenize()` collapses hyphens, and `catalog`
instead of `product catalog`. `target-fp-growth-affinity-model` therefore
falls from #2 to #7 on one surviving token, and `catalog` alone revives the
v1 false positive: `paypal-rule-governance-analytics` at #5 on fraud *rules*,
not products. Re-run this block after the tokenizer change.

**Keyword v5, brief rev 1** (28 must-haves: `catalog` → `catalog data`, plus
`catalog optimization` and `evaluation frameworks` — 2026-09-20). Scores
recorded 2026-09-23 with
`python -m src.tools.evidence --brief data/sample_jd.brief.yaml -k 8`:

```
1.  30.0  paypal-fra-insight-store               covers: Python, automation, knowledge graph, LLMs
                                                 supports: retrieval-augmented generation (RAG)
2.  17.0  target-goa-dmo-mismatch                covers: SQL, data analytics
3.  17.0  target-cross-functional-communication  covers: data analytics
4.  16.5  target-fp-growth-affinity-model        covers: recommendations, catalog data, catalog optimization
5.  12.5  self-text2sql-memory-agent             covers: SQL, Python, LLMs
6.  12.0  paypal-rule-governance-analytics       covers: SQL, Python
7.  11.5  pub-cyberbully-1dcnn-lstm              covers: classification, embeddings
8.   9.5  self-attrition-prediction              covers: Python, classification
```
Recall@5 = 4/5, back from 3/5, **before** any tokenizer change: the brief
revision did it. The coverage lines confirm the mechanism: `catalog data`
and `catalog optimization` are full matches for the FP-Growth atom
(#7 → #4), and with bare `catalog` gone `rule-governance-analytics` covers
only SQL and Python. The v1 false positive has left the top 5.

**Keyword v6** — hyphens collapsed, alias table (2026-09-23):

```
1.  30.0  paypal-fra-insight-store               covers: Python, automation, knowledge graph, LLMs
                                                 supports: retrieval-augmented generation (RAG)
2.  20.0  target-goa-dmo-mismatch                covers: SQL, eCommerce, data analytics
3.  19.5  target-fp-growth-affinity-model        covers: eCommerce, recommendations, catalog data, catalog optimization
4.  17.0  target-cross-functional-communication  covers: data analytics
5.  12.5  self-text2sql-memory-agent             covers: SQL, Python, LLMs
6.  12.0  paypal-rule-governance-analytics       covers: SQL, Python
7.  11.5  pub-cyberbully-1dcnn-lstm              covers: classification, embeddings
8.   9.5  self-attrition-prediction              covers: Python, classification
```
Recall@5 = 4/5, unchanged. What v6 changed: `eCommerce` is now covered by
both atoms tagged `e-commerce` (+3.0 each), FP-Growth moves #4 → #3, and
covered must-haves go from 11 to 12 of 28. The fifth expected atom (a
classification one) is still #7-8, behind `target-cross-functional-communication`
at #4 on `data analytics` alone. That is a ranking question for the retrieval
experiment, not a spelling one. Sixteen must-haves remain uncovered (taxonomy,
ontology, prompt engineering, model evaluation, search, semantic search,
entity resolution, …) — true gaps for the Judge.

---

## 002 — eBay, Data Scientist (Analytics), eBay Live (`data/ebay_live_analytics.brief.yaml`)

An experimentation / product-analytics role: a different axis from 001,
which was LLM/knowledge-graph heavy. Tests whether the brief conventions
and the v5 scorer hold on a second JD with no further tuning.

**Expected in top 5:** `target-goa-dmo-mismatch` (SQL, transactional data,
e-commerce), `target-fp-growth-affinity-model` (e-commerce,
recommendations), any Target experimentation / A/B / KPI atoms
(_fill in ids_), `paypal-rule-governance-analytics` (dashboards /
decision-health analytics).

**Keyword v5** (no changes since 001), before profile additions.
**Pre-alignment brief (13 must-haves):**

```
1.  19.2  target-goa-dmo-mismatch                  covers: SQL, product analytics
2.  15.0  paypal-rule-governance-analytics         covers: SQL, Python, KPIs
3.  12.7  target-cross-functional-communication    covers: product analytics
4.   8.5  target-fp-growth-affinity-model          covers: -   supports: e-commerce, recommendations
5.   7.5  target-miami-sort-center-delivery-radius covers: SQL
6.   6.8  self-text2sql-memory-agent               covers: SQL, Python
7.   6.8  pub-unet-vgg16-brain-tumor               covers: segmentation   <- polysemy: image, not customer
8.   5.8  paypal-fra-insight-store                 covers: Python
```
Uncovered: A/B testing, experimental design, hypothesis testing, causal
inference, transactional data, funnel analysis, dashboards,
instrumentation. Diagnosis: not a scorer problem — the profile has no
experimentation atoms although the CV lists those skills. Actions: add
experimentation atoms; tag GOA/DMO with `transactional data`; brief says
`customer segmentation`.

**Keyword v5, aligned brief** (27 must-haves; 2026-09-20):

```
1.  26.2  target-goa-dmo-mismatch                  covers: SQL, product analytics   supports: e-commerce
2.  21.2  target-cross-functional-communication    covers: product analytics
3.  18.0  paypal-rule-governance-analytics         covers: SQL, Python, KPIs
4.  10.8  target-fp-growth-affinity-model          covers: -   supports: e-commerce, recommendations
5.   9.0  paypal-fra-insight-store                 covers: Python   supports: automation
6.   8.0  self-text2sql-memory-agent               covers: SQL, Python   supports: AI
7.   7.5  target-miami-sort-center-delivery-radius covers: SQL
8.   7.2  pub-unet-vgg16-brain-tumor               covers: segmentation
```
Same shape as before: the fourteen new experimentation must-haves cover
nothing, because the profile still has no experimentation atoms. The brief
no longer says `customer segmentation`, so the brain-tumour publication is
now *covered* by `segmentation` rather than merely scored by it — the
polysemy is back in the open, where the retrieval experiment can measure it.

**Keyword v6** (2026-09-23; brief rev 1, which is what
`data/ebay_live_analytics.brief.yaml` holds). Identical to v5 on this brief,
atom for atom and score for score:

```
1.  26.9  target-goa-dmo-mismatch                  covers: SQL, product analytics   supports: e-commerce
2.  21.9  target-cross-functional-communication    covers: product analytics
3.  19.5  paypal-rule-governance-analytics         covers: SQL, Python, KPIs
4.  10.8  target-fp-growth-affinity-model          covers: -   supports: e-commerce, recommendations
5.  10.0  paypal-fra-insight-store                 covers: Python   supports: automation
6.   8.0  self-text2sql-memory-agent               covers: SQL, Python   supports: AI
7.   7.5  target-miami-sort-center-delivery-radius covers: SQL
8.   7.2  pub-unet-vgg16-brain-tumor               covers: segmentation
```
Expected: this brief spells `e-commerce` the way the atoms do, and the one
term the alias table fixes, `A/B testing` (now `['ab', 'testing']`), is
carried by no atom. Covered must-haves: 5 of 27 under both tokenizers.
Uncovered includes `transactional data`, so the GOA/DMO tag planned above
has not landed in `profile.yaml` yet. Rev 2's `causal measurement` would
change nothing here: same top eight, same scores, and no atom covers it. The
gap is data, not the scorer: see open item 7.

**Keyword v6, after profile additions:** _(paste ranking here)_

---

## Brief alignment (2026-09-20)

The hand-written briefs predate the Extractor's prompt rules, so the eval was
instructing one thing and grading another: the model split bundles, kept list
items and used the JD's spelling as told, and lost precision for it. Both
briefs were rewritten to follow the same rules. Each file's header documents
which rule produced which choice.

| case | must_haves | nice_to_haves | soft_skills |
|---|---|---|---|
| 001 sample_jd | 17 → 26 | 3 → 1 | 3 → 5 |
| 002 ebay_live_analytics | 13 → 27 | 8 → 11 | 3 → 5 |

What changed, by rule:

* **JD spelling** — `e-commerce` → `eCommerce`; `funnel analysis` →
  `funnels`; `customer segmentation` → `segmentation`; `product catalog` →
  `catalog` (trailing generic noun dropped from "catalog data").
* **Required *and* preferred is a must-have** — LLMs and prompt engineering
  moved to must_haves in 001; RAG is the only nice-to-have left there.
* **Every list member is an item** — 001 gained data wrangling, automation,
  search, structured data, taxonomy validation, prototyping, hypothesis
  testing. 002 gained statistics, behavioral data, event-level data,
  experimentation, success metrics, guardrails, controlled experiments,
  quasi-experimental methods, cohorts, behavioral analysis, measurement
  frameworks, self-service analytics, event tracking, data quality.
* **Bundles split** — `AI automation` → `AI` + `automation`.
* **Soft skills completed** — communication added to both.

Deliberately excluded, and now the reference for prompt v3: product areas and
buyer-journey stages (discovery, viewing, engagement, bidding, purchasing,
retention), manner rather than skill ("statistical rigor", "clear decision
criteria"), role categories on the years line ("data science"), and section
headings ("Product Analytics Attitude", "Startup DNA").

Dropped workarounds: `customer segmentation` existed to dodge the image-vs-
customer polysemy and `transactional data` was chosen because the evidence
covered it. Both are retrieval concerns, not descriptions of the job, so they
no longer shape the brief.

### Revisions after the multi-run evals

Revisions are numbered per case, counting from the aligned brief (rev 0).
Each change applies an existing rule to an item the alignment missed. The
model's output prompted the second look; the rule decided the change. An item
the model produces that no rule supports (journey stages, `discovery`) stays
out however often it appears.

| rev | date | case | change | rule | model had it |
|---|---|---|---|---|---|
| 1 | 09-20 | 002 | + `advanced experimentation`, `analytics workflows` (nice) | every list member on the Preferred lines | 3/3 each (M1) |
| 1 | 09-20 | 001 | `catalog` → `catalog data`; + `catalog optimization`, `evaluation frameworks` | a qualified generic word stays; every list member | 6/6 each (M2 + M4) |
| 2 | 09-22 | 002 | + `causal measurement` | fourth member of "statistics, experimental design, hypothesis testing, and causal measurement" | 9 of 14 (M1, M3, M6, M7) |

`causal measurement` reverses a call made during alignment and repeated under
E3 → E6 below, where it was treated as a near-duplicate of `causal inference`.
Rule 8 covers a longer or shorter form of *one* term ("forecasting" / "demand
forecasting"), not two different terms, so no rule supported the merge.

Current sizes: 001 has 28 / 1 / 5 (must / nice / soft), 002 has 28 / 13 / 5.
Row M7 (case 002) was scored before 002's rev 2; the next case-002 row is
the first against it.

*(2026-09-23: rev 2 is recorded here but was never saved to the file.
`data/ebay_live_analytics.brief.yaml` still has 27 must-haves, last edited
2026-09-21 20:05 IST, so it is rev 1. Everything scored against the file,
including the keyword v6 rescore, is rev 1 until it lands; see open items.)*

---

## Extractor runs (Phase 4)

P/R = must-have precision / recall against the hand-written brief; overlap =
top-5 retrieval Jaccard between the two briefs. Prompts: **v0** original ·
**v1** bucketing rule (nice_to_haves = explicitly optional only) · **v2**
split / list / generic-word rules, self-check, leak-free examples, title
check in `verify()` · **v3** product areas and years-line role names are not
items, activity-phrased requirements count, a years-to-seniority mapping,
more droppable tails; rule 4 checked in `verify()` (a violation costs a
retry) · **v3.1** rule 4 moved to `repair()` (fixed in code, no retry);
rule 6 gains "defining metrics, targets or thresholds is a requirement";
the client retries a rate limit up to 6 times.

| run | case | model | prompt | brief | P | R | overlap@5 | tokens in/out |
|---|---|---|---|---|---|---|---|---|
| E1 | sample_jd | gpt-oss-120b | v0 | pre-align (13) | 0.46 | 0.46 | 0.43 | 1856/1092 |
| E2 | sample_jd | gpt-oss-120b | v1 | pre-align (17) | — | 0.65 | 0.43 | — |
| E3 | ebay_live_analytics | gpt-oss-120b | v2 | pre-align (13) | 0.35 | 0.85 | 0.67 | 2532/1963 |
| E4 | sample_jd | gpt-oss-120b | v2 | pre-align (17) | 0.50 | 0.82 | 1.00 | 2591/2381 |
| E5 | sample_jd | gpt-oss-120b | v2 | pre-align (17) | 0.43 | 0.76 | 0.67 | 2591/2185 |
| E6 | ebay_live_analytics | gpt-oss-120b | v2 | aligned (27) | 0.92 | 0.85 | 1.00 | 2532/2081 |
| E7 | sample_jd | gpt-oss-120b | v2 | aligned (26) | 0.59 | 0.50 | 0.43 | 2591/1742 |

E2's P and token counts were not recorded at the time.

**E1 → E2, the bucketing rule.** nice_to_haves recall 0.14 → 1.00 with no
misplacements. My prompt said responsibilities-derived items were must_haves
while the brief said the opposite; the model followed the prompt and was
graded against the brief. A prompt-vs-reference conflict, not a model error —
the first instance of the pattern the alignment above finally removed.

**E2 → E4, prompt v2.** Recall 0.65 → 0.82 and overlap 0.43 → 1.00.
"eCommerce search" was split at last. Precision fell to 0.50, almost entirely
on items the reference was missing.

**E3 → E6, brief alignment.** Precision 0.35 → 0.92 on case 002, overlap
0.67 → 1.00, with 2 extras and 4 misses out of 27. Confirms the diagnosis:
most "extras" were gaps in the reference. The two survivors are the
judgment calls listed above — `data science` (role category) and
`causal measurement` (near-duplicate of causal inference) — both of which
prompt v3 should rule out explicitly. *(Later: `data science` was ruled out
by v3's years-line rule. `causal measurement` went the other way, into the
brief; see "Revisions after the multi-run evals".)*

**E4 vs E5 vs E7: run-to-run variance is the dominant effect.** Same prompt,
same model, same JD, `temperature=0`:

| run | P | R | overlap | completion tokens | what broke |
|---|---|---|---|---|---|
| E4 | 0.50 | 0.82 | 1.00 | 2381 | LLMs, prompt engineering, RAG absent entirely |
| E5 | 0.43 | 0.76 | 0.67 | 2185 | RAG and model evaluation absent |
| E7 | 0.59 | 0.50 | 0.43 | 1742 | "eCommerce search" bundled again; "classification models", "recommendation models"; LLMs and prompt engineering back in nice_to_haves |

Output quality tracks completion tokens: the shortest run broke rules the
longer ones followed. A hosted reasoning model is not deterministic at
temperature 0, and Groq's own reasoning guide says to benchmark across
multiple runs. **Consequence: a single run cannot support a claim about a
prompt change.** Differences under ~0.1 are noise until `extract_eval`
averages repeats.

E7's collapse is this effect, not the alignment: case 002 (E6) used the same
prompt against an aligned brief and scored 0.92.

### Multi-run rows

Identical inputs within a row; scores are mean (min-max). All rows are
`openai/gpt-oss-120b` on Groq. Brief sizes are must-haves (002 also
nice-to-haves); 002's must-haves are identical in rev 0 and rev 1, so its
must-have scores compare across M1, M3, M6 and M7. Case 001's do not compare
across rev 0 and rev 1 (26 vs 28 items).

| run | case | prompt | brief | runs ok | P | R | overlap@5 | attempts | tokens in/out (mean) |
|---|---|---|---|---|---|---|---|---|---|
| M1 | ebay_live_analytics | v2 | rev 0 (27/11) | 3/3 | 0.86 (0.81-0.96) | 0.94 (0.93-0.96) | 1.00 | — | 2532/2056 |
| M2 | sample_jd | v2 | rev 0 (26) | 3/3 | 0.71 (0.61-0.78) | 0.73 (0.65-0.81) | 0.53 (0.25-0.67) | — | 2591/2038 |
| M3 | ebay_live_analytics | v2 | rev 1 (27/13) | 3/3 | 0.92 (0.88-0.96) | 0.86 (0.81-0.93) | 1.00 | — | 4372/3006 |
| M4 | sample_jd | v2 | rev 0 (26) | 3/3 | 0.68 (0.67-0.68) | 0.67 (0.58-0.77) | 0.59 (0.43-0.67) | — | 2591/1953 |
| M5 | sample_jd | v3 | rev 1 (28) | 3/3 | 0.76 (0.64-0.82) | 0.67 (0.57-0.79) | 0.89 (0.67-1.00) | 2/2/2 | 5853/3970 |
| M6 | ebay_live_analytics | v3 | rev 1 (27/13) | 2/3 | 0.91 | 0.78 | 1.00 | x/2/2 | 5762/2767 |
| M7 | ebay_live_analytics | v3.1 | rev 1 (27/13) | 6/6 | 0.97 (0.95-1.00) | 0.91 (0.78-0.96) | 1.00 | 2/2/1/1/1/2 | 4316/2956 |
| M8 | sample_jd | v3.1 | rev 1 (28) | 6/6 | 0.85 (0.77-0.91) | 0.72 (0.64-0.79) | 0.78 (0.67-1.00) | 1/1/1/1/1/1 | 2861/2234 |
| M9 | ebay_live_analytics | v3.1 | rev 1 (27/13) | 6/6 | 0.95 (0.90-0.96) | 0.95 (0.89-0.96) | 1.00 | 1/2/1/2/1/2 | 4316/3511 |
| M9 | sample_jd | v3.1 | rev 1 (28) | 6/6 | 0.76 (0.69-0.83) | 0.72 (0.64-0.82) | 0.78 (0.67-1.00) | 1/1/1/1/1/1 | 2861/2691 |

`x` = a run that failed after its retry. The findings below run in order:
M1/M2 first, then the noise floor (M3/M4), v3 (M5/M6), and v3.1 (M7/M8).

#### M1, M2 — first multi-run baseline

**Single runs were unrepresentative in both directions.** E6 scored case 002
at P 0.92 against a mean of 0.86; E7 scored case 001 at R 0.50 against a mean
of 0.73. Neither conclusion drawn from them would have survived.

**Noise is a property of the JD, not of the pipeline.** Case 002 returned the
same five atoms in all three runs (overlap 1.00, recall range 0.03). Case 001
ranged 0.25-0.67 on overlap and pulled eight different atoms across three
runs. The difference is how the JDs are written: 002 has an explicit
"Required Qualifications" list, 001 is heading-and-dash marketing prose
("Analyze & Optimize eCommerce Product Data -"), where almost every
requirement has to be inferred from a phrase. Expect real postings to sit
across this whole range.

**Stated rules the model follows only sometimes.** The count is the number of
runs out of three, and it is the argument for moving a rule from the prompt
into `verify()`, where the retry enforces it. *(This was the v3
specification. What each rule became, and whether it now holds, is the
"Rule compliance, v2 → v3.1" table under M7, M8.)*

| rule | evidence | where it should live |
|---|---|---|
| no bare generic word | `analytics` 2/3, `metrics` 2/3 (001) | `verify()` — a closed word list |
| drop the trailing generic head | `evaluation frameworks` 3/3, `catalog optimization` 3/3, `product knowledge models` 2/3, `AI/ML models` 1/3, `AI-driven approaches` 1/3, `knowledge graph development` 1/3, `recommendation models` 1/3 (001) | `verify()` — a closed suffix list |
| product areas are not requirements | `discovery` 3/3 (001); `discovery`, `viewing`, `engagement`, `bidding`, `purchasing`, `retention` all 2/3 (002) | prompt v3 |
| split domain from capability | `eCommerce` extracted 1/3, `eCommerce search` 2/3 (001) | prompt v3 |
| required *and* preferred is a must-have | LLMs 2/3, prompt engineering 2/3 (001) | prompt v3 |
| name the object, not the activity | `catalog` 0/3 vs `catalog optimization` 3/3 (001) | prompt v3 |

Six of the thirteen case-001 extras end in a generic head noun. That is a
string check, not a judgement, which is the clearest signal yet that these
two rules belong in code.

**Reference gaps confirmed at 3/3.** Three runs agreeing against the brief is
the brief being wrong, not the model drifting. *(Overstated — see M3, M4
below: `analytics workflows` went 3/3 → 0/3 on the next batch. The rule
decided these additions; the 3/3 counts were weak support for them.)*

* `analytics workflows` (002, Preferred line) — missing from the brief; add.
* `early-stage product environment`, `rapidly scaling product environment`
  (002) — the brief dropped "environment"; either add it or add "environment"
  to the prompt's droppable generic heads.
* `advanced experimentation methodologies` 3/3 while the brief's must-have
  `experimentation` is never extracted (0/3) — the brief promoted a Preferred
  item to a must-have on the strength of one intro sentence. The model's
  reading is defensible; decide.
* `analytical problem solving` 3/3 against the brief's
  `analytical problem-solving` — a hyphen, nothing more. The tokenizer change
  removes it.

**Unresolved by either side.** `seniority` matched 0/3 on case 002: a
consistent disagreement, not noise. The JD's title carries no level and the
range is 4-8 years. Prompt v3 should state a concrete mapping (years floor to
level) instead of asking the model to infer one. *(Done in v3: matched in
every scored run since, 17 of 17 across both cases.)*

**Never extracted on case 001:** `catalog`, `experimentation`,
`hypothesis testing`, `prototyping` — all four come from the
"Prototype and Experiment - Move fast, test hypotheses" bullet, where the
requirement is phrased as an activity inside a working-style sentence. Either
prompt v3 teaches that shape, or the brief concedes those items. Left as a
measured gap for now.

#### M3, M4 — the noise floor on an unchanged reference

Same prompt (v2) and, for must-haves, the same briefs as M1 and M2, so every
must-have difference is sampling noise: case 002 P 0.86 → 0.92, R 0.94 →
0.86; case 001 P 0.71 → 0.68, R 0.73 → 0.67. Pooled over six runs, the v2
baseline is **002: P 0.89, R 0.90, overlap 1.00; 001: P 0.70, R 0.70, overlap
0.56**.

The item counts moved as much as the scores. `analytics workflows` went 3/3 →
0/3 and the six buyer-journey stages 2/3 → 0/3. An item that appears about
half the time shows 3/3 in one batch in eight by chance alone. **From here on
items are judged on pooled runs; only 6/6 or 0/6 counts as settled.**

M3's input tokens rose to 4372 per run from M1's 2532: retries, which v2 did
not report. The likely cause is visible in M7: a soft-skill grounding
rejection (`collaboration`) that costs a whole second call.

#### v3 — M5, M6

Rule 4 became `verify()` rejections, so a violation cost a retry.

* Detection worked. Every case-001 run was rejected once (`metrics` 3×,
  `analytics` 2×, `recommendation models` 2×) and fixed on the retry.
* The retry did not always work. On case 002 run 1 the model still returned
  `early-stage product environment` after being told to write
  `early-stage product`, and the run failed. On case 001
  "keep one" made it keep `recommendation models` and drop `recommendations`
  (0/3).
* Cost roughly doubled: ~5.8k tokens in per extraction against ~2.6k.
* Seniority matched in every run (0/6 under v2 on case 002): the mapping
  works.
* `KPIs` and `success metrics` fell to 0/2 on case 002 (5/6 under v2). The
  new product-area rule ("outcomes the business wants are context") was being
  stretched to cover *measuring* outcomes.

**Lesson: reject vs. repair.** All three rule-4 fixes are mechanical and
subtractive (drop a word, strip a tail, drop "X models" and keep "X"), so
code applies them and cannot introduce a claim. Retries stay for problems
whose fix needs judgement: grounding, title, years. That is v3.1.

#### v3.1 — M7, M8: the Extractor is accepted

**12 of 12 runs succeeded.** Title, company, seniority and years matched in
every run on both cases. Case 001 needed one call per extraction.

| | 002 v2 (M1+M3) | 002 v3.1 (M7) | 001 v2 on rev 1 (est.) | 001 v3.1 (M8) |
|---|---|---|---|---|
| must-have P | 0.89 | 0.97 | ~0.78 | 0.85 |
| must-have R | 0.90 | 0.91 | ~0.73 | 0.72 |
| nice-to-have P / R | 0.71 / 0.64 (M3) | 0.99 / 0.97 | — | — |
| overlap@5 | 1.00 | 1.00 | — | 0.78 |

The 001 v2 column is an estimate: re-scoring M2 + M4 against rev 1 turns the
two items the model produced 6/6 (`catalog optimization`, `evaluation
frameworks`) from extras into hits. It was not measured, because the v2
extractor had been replaced when rev 1 landed.

**Every gain since v2 is precision, and it comes from code.** On case 001 the
repairs dropped `analytics` 6/6, `metrics` 5/6 and `recommendation models`
2/6; on case 002 they stripped eleven `…methodologies` / `…environment` tails
and one `…use cases`, which is most of the nice-to-have jump. Recall has not
moved on either case since v2.

**Retrieval — the number that matters downstream.** Case 002 returned the
same top five in all 14 case-002 runs since v2 (M1, M3, M6, M7). On case 001
the extracted brief pulled `paypal-fra-insight-store` and
`target-fp-growth-affinity-model` in 6/6. The disagreement is mostly the fifth
slot: `self-text2sql-memory-agent` 4/6 against `pub-cyberbully-1dcnn-lstm`
4/6, which sat a point apart (12.5 vs 11.5) in the last recorded 001 ranking.

**Rule compliance, v2 → v3.1.** Pooled counts; each rule and what now
enforces it:

| rule | v2 (6 runs) | v3.1 (6 runs) | enforced by | verdict |
|---|---|---|---|---|
| no bare generic word | `analytics` 5/6 (001) | emitted 6/6, repaired 6/6 | `repair()` | holds, in code |
| drop a generic tail | three tailed items 18 of 18 (002) | emitted 11 of 18, all repaired | `repair()` | holds, in code |
| "X" and "X models" are one item | `recommendation models` 3/6 (001) | repaired 2/6 | `repair()` | holds, in code |
| seniority mapping | 0/6 (002) | 6/6 both cases | prompt | holds |
| defining metrics is a requirement | KPIs, success metrics 5/6 (002) | 4/6 each; guardrails 2/6 → 4/6 | prompt | recovered from v3's 0/2 to about the v2 level |
| journey stages are not items | 2/6 (002) | 0/6 | prompt | holds |
| product areas are not items | `discovery` 6/6 (001) | 5/6 | prompt | fails — rules 2 and 6 conflict: the JD's "search & discovery" is a list |
| split a domain from a capability | `eCommerce` 2/6 (001) | 1/6 | prompt | fails |
| required *and* preferred is a must-have | LLMs misplaced 2/6 (001) | 3/6 | prompt | fails |
| activity-phrased requirements count | `experimentation`, `hypothesis testing`, `prototyping` 0/6 each (001) | 1/6, 1/6, 0/6 | prompt | fails |
| "wherever it appears" includes the intro | `data analytics` 1/6 (001), `experimentation` 0/6 (002) | 0/6, 0/6; `catalog data` 0/6 (001) | prompt | fails |

The split is clean. Rules that are string checks now live in code and hold.
Prompt rules that are local and concrete — a mapping, a named exception —
hold. Prompt rules that need the model to read the JD's structure (which
paragraph, which list, which sentence also appears under Preferred) hold
half the time or less, and prompt wording has stopped moving them. **The
five failing rules account for most of what is left — four of them are
recall, `discovery` is precision — and they are the test set for the
end-of-phase model comparison.**

**Cost.** Case 001 is back to about one call per extraction (2861/2234
tokens). Case 002 averages 4316 in because 3 of 6 runs retried on one
soft skill, `collaboration`: the JD says "working effectively with
cross-functional … teams", which the lenient grounding check does not
connect to that word. Soft skills feed phrasing, not retrieval, so a retry
there buys little.

**Surface-form mismatches** — same requirement, different spelling, scored as
a miss plus an extra:

| pair | cause | status |
|---|---|---|
| "retrieval-augmented generation (RAG)" vs "…generation" | parenthesised acronym | fixed in `_key()` |
| "analytical problem-solving" vs "analytical problem solving" | spacing — the v6 tokenizer alone still gives `problemsolving` vs `problem solving` | fixed in `_key()`: spacing ignored (2026-09-23); was a soft-skill miss + extra in 5/6 (M7) |
| "eCommerce" (brief) vs "e-commerce" (atom tags) | hyphen inside a word | fixed in `tokenize()`, keyword v6 (2026-09-23) — a retrieval mismatch, not an eval one |
| "rapidly scaling product" vs "rapid scaling product" | adverb/adjective inflection | undecided — stemmer, or accept; not seen since E6 |
| "retrieval-augmented generation (RAG)" vs "RAG" | acronym *alone*, no expansion to strip | open — `compare()` could match an item to the initials of another; 3/6 (M8) |

#### Keyword v6 — M7, M8 re-scored (2026-09-23)

`python -m src.tools.extract_eval --rescore` re-scores the twelve saved
M7/M8 briefs (`data/evals/extracted/*.r1-r6.brief.yaml`) with no API calls.
Scoring is deterministic, so any difference between rows is the scorer or
the reference, never sampling.

| scorer | case (brief) | must-have P / R | nice P / R | soft P / R | overlap@5 |
|---|---|---|---|---|---|
| v5 tokenizer | 002 (rev 1) | 0.97 / 0.91 | 0.99 / 0.97 | 0.56 / 0.33 | 1.00 |
| v6 tokenizer | 002 (rev 1) | 0.97 / 0.91 | 0.99 / 0.97 | 0.56 / 0.33 | 1.00 |
| v6 + spacing-free `_key()` | 002 (rev 1) | 0.97 / 0.91 | 0.99 / 0.97 | **0.82 / 0.50** | 1.00 |
| v6 + spacing-free `_key()` | 002 (rev 2, not yet in the file) | 0.99 / 0.90 | 0.99 / 0.97 | 0.82 / 0.50 | 1.00 |
| v5 tokenizer | 001 (rev 1) | 0.85 / 0.72 | 0.72 / 0.83 | 0.74 / 0.53 | 0.78 (0.67-1.00) |
| v6 tokenizer | 001 (rev 1) | 0.85 / 0.72 | 0.72 / 0.83 | 0.74 / 0.53 | 0.78 (0.67-1.00) |
| v6 + spacing-free `_key()` | 001 (rev 1) | 0.85 / 0.72 | 0.72 / 0.83 | 0.74 / 0.53 | 0.78 (0.67-1.00) |

**The v5 rows reproduce M7 and M8 exactly**, which checks `--rescore` and
identifies the saved files as that batch. It also shows M7 was scored
against the file as it still is: rev 1.

**Rev 2** was scored on a temporary copy of the brief with `causal
measurement` added, because the file does not have it yet. The item was
produced in 4/6 runs, so it turns from an extra into a hit: P 0.97 → 0.99,
R 0.91 → 0.90 (28 expected now, and two runs missed it).

**The tokenizer alone moved no eval number.** The model copies the JD's
spelling and so do the references (rule 5), so an extracted item and its
reference never differed by a hyphen. v6 matters where two *different
writers* meet — a brief written from the JD against atoms written by hand —
which is retrieval (case 001's `eCommerce` above), not this comparison.

**The soft-skill fix needed the key, not the tokenizer.** "analytical
problem-solving" vs "analytical problem solving" is a spacing difference: v6
turns it into `problemsolving` vs `problem solving`, still two keys. `_key()`
now joins tokens without spaces. It lives in the eval, not in `tokenize()`,
because retrieval needs word tokens. Audited against all 103 distinct items
in the references and saved runs: the only merge it makes is this one. Case
002 soft skills P 0.56 → 0.82, R 0.33 → 0.50; `analytical problem-solving` is
now hit 5/6. The two remaining soft misses, `clear product recommendations`
and `structuring ambiguous questions` (0/6 each), are paraphrased by the
model ("structured thinking"). That is meaning, not spelling.

**Overlap is unchanged** because the top-5 *sets* are. Only the order inside
case 001's reference top 5 moved (FP-Growth above cross-functional).

Not measured here: `repair()` and `verify()` also use the tokenizer, and
under v6 they treat "e-commerce" and "eCommerce" as one item: deduplicated
within a bucket, rejected as a cross-bucket duplicate. That shows only in a
fresh API run. *(M9, below, is that fresh run.)*

#### Model comparison — M9 and Qwen 3.8 27B (2026-09-30): gpt-oss-120b stays

The plan: the same eval on `qwen/qwen3.8-27b` against a fresh gpt-oss-120b
batch, judged on the five failing prompt rules and on recall. Before
running it, every request started stating its settings
(`config.MODEL_SETTINGS`): Groq's defaults differ per model, and for Qwen
the default reasoning effort is `none` (no thinking at all), which would
have compared gpt-oss thinking against Qwen not thinking. Settings used:
gpt-oss effort medium, temperature 0 (unchanged from M7/M8, whose default
was already medium); Qwen effort medium, reasoning hidden, temperature 1.0,
top_p 0.95 (Groq's recommended thinking-mode sampling). Batches are saved
under `--tag m9`, so M7/M8's files are untouched.

**M9 — gpt-oss-120b, fresh** (rows above). 12 of 12 runs succeeded. It is
the first run with keyword v6 inside `repair()` and `verify()`.

* Case 002: P 0.97 → 0.95, R 0.91 → 0.95 against M7. One or two items of 27
  either way; still 3 of 6 runs retried on `collaboration`.
* Case 001: P 0.85 → 0.76, R unchanged at 0.72 against M8. About two extra
  items per run (`discovery` 5/6, `product data` 4/6, `eCommerce search`
  3/6). The ranges overlap (0.69-0.83 vs 0.77-0.91), and case 001 has always
  been the noisy JD (see M1-M4), so this is read as sampling, not a
  regression.
* Retrieval overlap identical to M7/M8: 1.00 and 0.78.
* The five failing rules, M8 → M9: `discovery` 5/6 → 5/6; `eCommerce` 1/6 →
  2/6; LLMs misplaced 3/6 → 3/6; activity-phrased items `experimentation` /
  `hypothesis testing` / `prototyping` 1/6, 1/6, 0/6 → 2/6, 5/6, 2/6; intro
  items still 0/6. Still failing; `hypothesis testing` is the only real
  movement and one batch cannot settle it.

**Qwen 3.8 27B on Groq — could not be run.** Groq's free tier refused most
requests with "Request too large", before running them:

| attempt | max_tokens | 002 runs ok | 001 runs ok | 002 P / R (ok runs) | 001 P / R | tokens in/out (mean) |
|---|---|---|---|---|---|---|
| 1 | 8192 | 3/6 | 0/6 | 0.93 (0.90-0.95) / 0.89 (0.74-0.96) | — | 2628/1260 |
| 2 | 4096 | 2/6 | 1/6 | 0.89 (0.82-0.96) / 0.78 (0.67-0.89) | 0.92 / 0.86 | 2628/1252 |

Attempt 2's refusals reported "limit 1000, requested 1200-1764", and the
requested size varied between identical requests. That matches neither
Groq's documented 8K tokens per minute for this model nor prompt plus
max_tokens, so the limit being enforced is not the documented one. (The
error line cut off which limit it named; it now keeps it.) Only attempt 2's
three successful briefs are saved
(`data/evals/extracted/*.qwen-qwen3-8-27b.m9.*`).

What the six successful Qwen runs show: on case 002, P 0.91 / R 0.85 pooled
over five runs, against gpt-oss's 0.95 / 0.95. That is not better. Qwen
used a third to a half of gpt-oss's output tokens. The single case-001 run
(0.92 / 0.86) is one sample, not evidence. The five-rule question is
unanswered.

**Decision: the Extractor stays on `openai/gpt-oss-120b` via Groq.** It is
accepted (M7/M8), reproduced (M9), and runs 12 of 12 on the free tier. A
fair Qwen comparison needs a provider that serves it reliably. Cerebras's
free tier serves both models (`gpt-oss-120b`, `qwen-3.8-27b`; 30K tokens a
minute), and `config.PROVIDERS` / `--provider` can already route there.
Deferred until extraction quality is the pipeline's bottleneck.

---

## Ranker runs (Phase 4)

**Pool.** "data scientist" in Bangalore, 3 JSearch pages, 28 postings
(fetched 2026-09-27). Labelled by hand: 12 yes, 5 maybe, 8 no, 3 skip. The
labels file (`data/jobs_cache/labels/`) is gitignored because it records
which jobs I would apply to, so postings are described here by level, never
by company.

**Metrics** (`src/tools/rank_eval.py`). P@5 strict = share of the top 5
labelled yes; lenient = yes or maybe. Skipped postings are removed before
scoring. Stage-1 recall@15 = share of the 12 yes postings that stage 1 keeps
in its top 15, the only postings stage 2 sees. Random pick: strict 0.48,
lenient 0.68, recall@15 0.54; best possible 1.00 / 1.00. Scale: one yes
posting = 0.08 recall, one top-5 slot = 0.20 P@5.

### Run 1 — MiniLM, before the stage-1 options (2026-09-27)

Stage 1: P@5 0.40 / 0.80, recall@15 0.42 (5 of 12 yes). Stage 1 + 2, three
runs: strict P@5 0.73 (0.60-0.80), scored before the `skip` label existed.
Of stage 2's 15 top-5 slots, 11 were yes postings and 4 were postings later
labelled skip.

**Stage 2 worked; stage 1 was the bottleneck.** Stage 2 can only reorder the
fifteen it is shown, and stage 1 had dropped 7 of the 12 yes postings before
it. What followed: the title counts as a sentence, atoms can be excluded,
optional centering, pluggable embedders, and the `skip` label.

### Stage-1 comparison (2026-09-29)

`rank_eval --compare`, stage 1 only, CPU torch. `-1 atom` excludes the
soft-skill atom, which matched every posting's collaboration line.

| embedder | stage-1 variant | recall@15 (yes) | P@5 strict | P@5 lenient |
|---|---|---|---|---|
| minilm | no titles | 0.42 | 0.40 | 0.80 |
| minilm | titles | 0.42 | 0.40 | 0.80 |
| minilm | titles, -1 atom(s) | 0.58 | 0.40 | 0.60 |
| minilm | titles, centered | 0.33 | 0.20 | 0.60 |
| minilm | titles, -1 atom(s), centered | 0.33 | 0.20 | 0.60 |
| bge-base | no titles | 0.50 | 0.40 | 0.60 |
| bge-base | titles | 0.50 | 0.40 | 0.60 |
| bge-base | titles, -1 atom(s) | 0.50 | 0.60 | 0.60 |
| bge-base | titles, centered | 0.42 | 0.20 | 0.40 |
| bge-base | titles, -1 atom(s), centered | 0.42 | 0.20 | 0.40 |
| qwen3-0.6b | no titles | 0.50 | 0.60 | 0.80 |
| qwen3-0.6b | titles | 0.50 | 0.60 | 0.80 |
| qwen3-0.6b | titles, -1 atom(s) | 0.50 | 0.60 | 0.80 |
| qwen3-0.6b | titles, centered | 0.50 | 0.60 | 0.80 |
| qwen3-0.6b | titles, -1 atom(s), centered | 0.58 | 0.60 | 0.80 |

Titles never moved a posting across rank 15. Centering hurt on MiniLM and
BGE and did nothing on Qwen. Qwen3-0.6B was the steadiest on P@5. **But the
best recall, 0.58 (7 of 12), barely beats a random cut (about 6.4 of 12)**,
and every difference in the table is one or two postings. `qwen3-4b` was not
run (not pulled in Ollama).

### Level, not topic

All 28 postings come from one "data scientist" search, so all are on topic
and similarity barely separates them. What separates the labels is the
experience asked. Years below are as `years_asked()` reads them: the largest
lower bound among the years phrases in the title and description.

| label | years asked |
|---|---|
| yes (12) | none stated ×5, 3, 4, 4, 4, 5, 6, 6 |
| maybe (5) | 3, 5, 6, 6, 8 |
| no (8) | 6, 7, 7, 8, 8, 11, 12, 12 |
| skip (3) | none stated, 2, 4 |

No yes posting asks for 7+ years; 7 of the 8 no postings do. With 5.2 years
of experience (resume.json), "demote anything asking more than 5.2 + 1.5 =
6.7 years" catches 7 no, 1 maybe (8+) and no yes. The one no it cannot catch
is a Principal role asking 6+.

(A manual read put one no posting at 11 years: "Experience: 7 11 Years". The
parser reads it as a 7-11 range whose dash the job board dropped, so 7.
Either value is demoted.)

### The level gate

After stage 1 scores the pool, each posting's minimum years (`years_asked()`)
is compared with the candidate's total experience from resume.json. Postings
asking more than that plus `level_slack` (default 1.5) move below all the
others, keeping their order, with a reason ("asks 8+ yrs (you: 5.2)"). They
are demoted, not dropped, because a text pattern will sometimes misread, and
a demoted posting still appears in `shortlist.json`. The demotion holds
through stage 2: the reranker orders postings within each tier.
`Stage1Options.level_slack`; `--no-gate` turns it off.

### Stage-1 comparison with the gate (2026-09-29)

Same command, two rows added per embedder. The five rows above reproduced
exactly, so the gate is the only change.

| embedder | stage-1 variant | recall@15 (yes) | P@5 strict | P@5 lenient |
|---|---|---|---|---|
| minilm | titles, level gate +1.5y | 0.83 | 0.60 | 1.00 |
| minilm | titles, -1 atom(s), level gate +1.5y | 0.83 | 0.60 | 1.00 |
| bge-base | titles, level gate +1.5y | 0.67 | 0.80 | 1.00 |
| bge-base | titles, -1 atom(s), level gate +1.5y | 0.67 | 0.80 | 1.00 |
| qwen3-0.6b | titles, level gate +1.5y | 0.83 | 0.80 | 1.00 |
| qwen3-0.6b | titles, -1 atom(s), level gate +1.5y | 0.83 | 0.80 | 1.00 |

* **The gate is the fix.** Recall 0.42-0.58 → 0.67-0.83 (2 to 4 more yes
  postings reach stage 2), and lenient P@5 is 1.00 on every embedder.
* **Qwen3-0.6B** ties MiniLM on recall and is one posting better on strict
  P@5. BGE matches its P@5 but keeps two fewer yes postings.
* **With the gate on, the atom exclusion changes nothing** on any embedder.
  Kept for its reason (a soft skill is phrasing, not evidence), but inert on
  this pool.

**Defaults chosen** (`ranker.default_options()`): Qwen3-0.6B, titles on, no
centering, the atoms in `RANKER_EXCLUDE_ATOMS` (`.env`) left out, level gate
+1.5 years.

### Acceptance run (2026-09-29)

`python -m src.tools.rank_eval <labels>` with the defaults; stage 2 on
`openai/gpt-oss-120b`, three runs.

| | P@5 strict | P@5 lenient | recall@15 (yes) |
|---|---|---|---|
| random pick | 0.48 | 0.68 | 0.54 |
| run 1, stage 1 (MiniLM) | 0.40 | 0.80 | 0.42 |
| stage 1 (Qwen3-0.6B + gate) | 0.80 | 1.00 | 0.83 (10/12) |
| **stage 1 + 2, 3 runs** | **0.93 (0.80-1.00)** | **1.00, every run** | — |

Tokens per run (mean): 5297 in / 4001 out.

* **The gate demoted 8: 7 no, 1 maybe, 0 yes**, exactly as the years table
  predicted.
* **No "no" posting reached the top 5 in any run.** The Principal role asking
  6+, the one no the gate cannot catch, never got there: stage 2 judges level
  from the title.
* **Stage 2 adds value on top of stage 1.** Four of the five top slots were
  the same in all three runs, all yes. Two of those four were not in stage
  1's top 5; stage 2 promoted them and pushed out a Director-titled maybe.
* **The 0.80-1.00 spread is one posting.** The fifth slot rotated between two
  yes postings and a maybe.
* Stage 1 still cuts 2 of the 12 yes postings before stage 2. This output
  does not say which.

**Accepted.** The bar: no "no" in the top 5 in any run (lenient P@5 1.00 in
every run); stage 1 + 2 at least as good as stage 1 alone (every run ≥ 0.80
strict, 1.00 lenient); tests green (198 passed).

**Caveat: this result is in-sample.** The gate's 1.5-year slack and the
embedder were both chosen by looking at these 28 labels, then measured on
them. The 1.00 shows the rule fits this pool, not yet that it predicts a new
one. Three things are unproven:

* The slack. Here it fell neatly between the yes postings (6 years at most)
  and 7 of the 8 no postings (7 years or more). Another pool may hold a
  7-year role worth applying to.
* The years pattern. It was written against these postings' phrasings. "A
  decade of experience" or "8 yrs min" would be missed or misread.
* Sample size. One posting moves P@5 by 0.20.

The check: fetch a fresh pool, label it *before* ranking, and rerun. Planned
for Phase 5, when the Sourcer fetches new postings anyway.

Known over-reads by `years_asked()`: alternatives read as the larger ("6+
years with a Masters or 4+ with a PhD" → 6), and a bare number next to
"experience" in company prose counts as a requirement. Demoting instead of
dropping is the hedge for both.

---

## Tailor + fit runs

| run | brief | attempts | tokens in/out | changes | uncovered | fit |
|---|---|---|---|---|---|---|
| 001a | sample_jd (2-page target, pre-fit) | 1 | 11009/4979 | 11 | 4 | 2 pages: 1 + orphan line |
| 001b | sample_jd (1-page target) | 1 | 11179/4352 | 10 | 4 | 0 steps, 1 page, fill 0.97 |
| 002 | ebay_live_analytics (1-page) | 1 | 10973/4910 | 8 | 9 | 0 steps, 1 page, fill 0.99 |

001a findings: soft-skill-only bullet (→ rule 6), keywords appended not
embedded, non-numeric phrases unverifiable by string check, skill items
inferred from summaries (→ normalised skill check). 002: nine uncovered
must-haves because the profile has no experimentation atoms — correct
behaviour, weak application.

Both runs predate the brief alignment; the briefs they consumed no longer
exist in that form.

---

## Pipeline runs (Phase 5)

`python run.py --keyword "data scientist" --location Bangalore --pages 3`,
on the pool cached 2026-09-27 (28 postings, no JSearch request). Stage 1 on
Qwen3-0.6B with the level gate, stage 2 and the Extractor on
`openai/gpt-oss-120b`.

| run | date | top | Tailor | delivered | time | ranker tokens in/out | extractor tokens in/out |
|---|---|---|---|---|---|---|---|
| P1 | 2026-11-16 | 1 | Claude (live) | 0/1: failed at `tailor` | ~11 min (stage 1 uncached) | 4656/3188 | 2854/3824 |
| P2 | 2026-11-16 | 3 | stubbed (returns the resume unchanged) | 3/3, one page each, fill 0.88 | 186 s (stage 1 cached) | 6530/5100 | 9117/9758 |

P1 is the first fully live run. It reached the Tailor and Anthropic answered
400, "Your credit balance is too low". Not a pipeline fault: the posting was
recorded as `failed` with `error.json` (stage, error, traceback, tokens
spent so far), `summary.json` was written and the run exited 1. **Phase 5's
acceptance, three tailored PDFs from one command, is still open.**

P2 replaces only the Tailor's Claude client, so every other node ran for
real on three postings: cached sourcing, both ranking stages, extraction
with grounding, retrieval, Chromium rendering, the fit loop and the Judge's
checks. The PDFs are the base CV; it says nothing about tailoring quality.

What the two runs show:

* **The shortlist's head is stable, its tail is not.** Ranks 1 and 2 were
  the same postings in both runs (9/10 each). Rank 3 differed. Same lesson
  as the acceptance run: one slot moves between stage-2 runs.
* **Extraction varies on one JD.** The rank-1 posting gave 73 must-haves in
  P1 and 81 in P2 (87 in the October dry run).
* **Briefs from real postings are large and mostly uncovered.**

  | posting (rank in P2) | must-haves | with evidence in the profile |
  |---|---|---|
  | 1 | 81 | 11 (14%) |
  | 2 | 26 | 7 (27%) |
  | 3 | 62 | 8 (13%) |

  The hand-written briefs have 27 and 28 must-haves. The Judge reported 79,
  26 and 64 gaps: nearly all are "no evidence in the profile".
* **Every posting asks for an expansion.** The base CV fills 0.88 of the
  page and the expand target is 0.95, so the fit loop called the (stubbed)
  Tailor once more per posting. Live, that is a second Claude call per CV
  before any tailoring effect on length.
* **Stage 1 is a one-off cost per pool.** ~10 minutes to embed the pool's
  sentences on CPU, 22 s once the vectors are in
  `data/jobs_cache/embeddings/`.

---

## Open items

Phase 5 built 2026-11-16; its live acceptance run is open (needs Anthropic
credit), as are the fresh-pool check and the retrieval experiment below.

Phase 4 closed 2026-10-05: deliverable 1 (`jobs.py`) done; deliverable 2
(the Extractor) accepted at prompt v3.1 on M7 + M8, reproduced by M9; the
tokenizer (keyword v6) done 2026-09-23; deliverable 3 (the Ranker) accepted
2026-09-29; the model comparison closed with gpt-oss-120b kept; deliverable
4 (the retrieval experiment) moved to Phase 5.

1. ~~**Tokenizer**~~ — done 2026-09-23. It restored `eCommerce` coverage in
   retrieval. Case 001's recall@5 had already been restored by brief rev 1,
   and the `analytical problem-solving` hit needed the spacing-free `_key()`,
   not the tokenizer. See "Keyword v6 — M7, M8 re-scored".
2. ~~**Ranker**~~ (deliverable 3) — accepted 2026-09-29 with the level gate;
   see "Ranker runs". Follow-ups:
   * **Fresh-pool check** (Phase 5): label a new pool before ranking it; the
     acceptance numbers are in-sample until then.
   * Optional: have `rank_eval` name the yes postings stage 1 cuts before
     stage 2 (2 of 12 on this pool). Free to run with `--stage1-only`.
3. **Retrieval experiment** (deliverable 4) — moved to Phase 5. Keyword v6
   vs embedding vs hybrid, on 10-15 hand-labelled cached JDs. It needs a
   labelling pass that does not block the graph, and the Phase 5 Sourcer
   produces the fresh postings it needs.
4. ~~**Model comparison**~~ — closed 2026-09-30, gpt-oss-120b kept; see
   "Model comparison — M9 and Qwen 3.8 27B". Optional later: Qwen on
   Cerebras (`--model qwen-3.8-27b`, needs `CEREBRAS_API_KEY`) against
   `--model gpt-oss-120b` on the same provider. Also untried: Groq's strict
   schema mode (`strict: true`), and moving the system prompt into the
   user turn, which Groq advises for reasoning models.
5. **Soft-skill grounding retry** (optional, cost only) — 3 of 6 case-002
   runs spent a whole second call on the soft skill `collaboration`. Dropping
   an ungrounded soft skill in `repair()` instead would be subtractive and
   save those calls.
6. **`compare()` and bare acronyms** — "RAG" alone still matches nothing
   (3/6 in M8). A second matching pass on initials would close it.
7. **Manual, any time:**
   * Case 002 rev 2 — add `causal measurement` to the must-haves in
     `data/ebay_live_analytics.brief.yaml` (the revision table says it
     landed; the file says it did not), then re-run `--rescore`. Expected:
     must-have P/R 0.99 / 0.90.
   * `profile.yaml` has no experimentation atoms, so every experimentation
     must-have in case 002 is uncovered — and `transactional data` is too,
     so the planned GOA/DMO tag is still missing. Add atoms, then fill the
     "after profile additions" block.
   * Case 002 expected-atom ids (`_fill in ids_` above), needed before the
     retrieval experiment can score recall@k on this case.
   * ~~Case 001 rev 1 ranking with scores~~ — recorded 2026-09-23.