# EPC LLM Labeling Prompt (step 27)

Each (query, article) pair is labeled by one LLM call: 12 queries × 100 articles = 1,200
calls. Every call uses the same system prompt. The user message is filled in per pair.
Placeholders are in `{CURLY_BRACES}`.

## System prompt (identical for every call)

```
You are a relevance assessor building a test collection for evaluating a search engine.
You judge whether a single news article substantively addresses a user's information
need, following the relevance guideline below exactly. You are careful, consistent and
literal: you base every decision only on the text of the article in front of you, never
on outside knowledge about the topic or the institution.

<guideline>
{GUIDELINE}
</guideline>
```

`{GUIDELINE}` is the full contents of `docs/relevance_guideline.md`, pasted in word for word.

## User message template (filled in for each pair)

```
<article>
<document_id>{DOCUMENT_ID}</document_id>
<title>{TITLE}</title>
<text>
{TEXT}
</text>
</article>

<information_need>
<query>{QUERY_TEXT}</query>
<description>{DESCRIPTION}</description>
</information_need>

Decide whether the article above is relevant to the information need, applying the
guideline. Judge against the description, not just the query words.

Respond with a single JSON object and nothing else (no markdown, no code fences, no
text before or after it), using exactly these fields:

{
  "label": 0 or 1,
  "reason": "1-2 sentences explaining the decision in terms of the guideline",
  "excerpt": "a passage copied word for word from the article, at most 50 words, or \"none\"",
  "uncertain": true or false
}

Rules for the fields:
- label: 1 = relevant, 0 = not relevant. No other values.
- reason: refer to what the article does or does not cover, not to the query words alone.
- excerpt: copy it exactly from the article text, with no paraphrasing, no added
  ellipses and no changes in punctuation. For label 1, give the passage that best shows
  the article addresses the need. For label 0, give the passage where the topic comes
  closest, or "none" if the topic does not appear at all.
- uncertain: true only if this is a borderline case where you could reasonably have
  given the other label. Otherwise false.
```

Placeholder sources:

| Placeholder | Source |
|---|---|
| `{DOCUMENT_ID}`, `{TITLE}`, `{TEXT}` | `document_id`, `title`, `text` in `articles.json` |
| `{QUERY_TEXT}`, `{DESCRIPTION}` | `text`, `description` in `queries.json` |

## Output format

Each call returns one JSON object, for example:

```json
{
  "label": 1,
  "reason": "The article explains the stages of the SU election, including the Grey Period, nominations and campaigning rules, and the Election Commission's role in enforcing them.",
  "excerpt": "The election process begins with the declaration of the Grey Period (July 25 to August 25). An essential 'cool-down period', it requires the candidates to stop campaigning and bars them from taking on any new initiatives.",
  "uncertain": false
}
```

The labeling script adds `query_id` and `document_id` and writes one record per pair to
`results/llm_labels.jsonl`. It then writes `qrels.csv` (`query_id,document_id,relevance`)
using the `label` field.

## Design choices

- **The guideline goes in the system prompt** so all 1,200 calls are judged by identical rules.
- **The article comes before the question.** With long articles (some are over 4,000
  words), models tend to judge more accurately when the document comes first and the task last.
- **The label comes before the reason**, which keeps the output short and easy to parse.
  Any change to field order must be made before the full run, not partway through.
- **Excerpts must be word for word** so the script can confirm each one appears in the
  article, which catches made-up evidence.
- **Outside knowledge is not allowed**, so an article is never marked relevant for what
  the model knows about BITS Pilani instead of what the article says.

## Run settings (fill in before the run)

- Model and exact version:
- Temperature: 0
- Date of run:

## Checks after the run

1. Every pair returned valid JSON with all four fields and a label of 0 or 1. Retry any that didn't.
2. Every excerpt other than `none` appears in its article. Normalise curly vs. straight
   quotes on both sides before comparing.
3. Review all `uncertain: true` pairs by hand, plus a random sample of about 50 others,
   and report how often you agree with the LLM.
4. Every query has at least one relevant article. A query with none is reported
   separately and left out of the averages (step 17).
