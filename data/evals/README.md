# Retrieval eval set

Hand-labelled cases for the Phase 4/5 retriever experiment (keyword vs
embedding vs hybrid, see ROADMAP). One entry per JD: the brief, what
*should* surface, and what each scorer actually did. Five minutes per
JD; the point is that "scorer X is better" becomes a measured recall@k
claim rather than an impression from one run.

Format per case: expected atoms (unordered, the ones a human would put on
the CV for this job), then one ranking block per scorer version.

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
them). **Phase 3 baseline.**

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

**Keyword v5** (no changes since 001), before profile additions:

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

**Keyword v5, after profile additions:** _(paste ranking here)_

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
