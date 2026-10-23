# EPC Relevance Guideline

### 1. The task
You are given an **information need** (a query and its description) and **one article** from the English Press Club collection. Decide whether the article is **relevant (1)** or **not relevant (0)** to that need. Judge each (query, article) pair on its own, without reference to other articles or other queries.

### 2. The core rule
> An article is **relevant** if it **substantively addresses** the information need. **Mentioning the query words is not enough.**

Judge against the **description**, not the query text. The query text is only the words a user typed; the description says what they actually want. An article can be relevant without using the query words, and can use them often without being relevant.

### 3. Relevant (label 1) if **any** of these hold
- A meaningful portion of the article (a section, or several connected paragraphs) is about the need.
- A reader with this need would **learn something useful** from the article that they didn't know before: facts, explanation, figures, rules, experiences or opinions on the topic.
- The article's main subject is something else, but it contains a **substantial, self-contained part** that addresses the need. For example, a Students' Union report card with a detailed section on sports facilities is relevant to the sports query.

### 4. Not relevant (label 0) if **any** of these hold
- The topic appears only as a **passing mention**, a single sentence, or an item in a list.
- The query words appear but in a **different sense or context** than the description means. For example, "budget" referring to a club's spending when the need is about fest budgets.
- The article is on a **neighbouring topic** that the description explicitly excludes. For example, internships when the need is placements, or other eateries when the need is the mess.
- The article only **names** the topic, or promises to discuss it, without actually giving information about it.

### 5. Edge cases
- **Duplicates and near-duplicates:** the collection has several articles with the same or similar titles (e.g. multiple "Packing for Pilani" and "Interview with the Election Commission" articles). Judge each one on its own content. Do not mark one as non-relevant because another covers the same ground.
- **Interviews and debates:** a Q&A or debate is relevant if the questions and answers on the topic are substantive, even if they are only one part of the conversation.
- **Outdated information:** an article from an earlier year is still relevant if it substantively addresses the need. Do not judge whether its information is current.
- **Opinion vs. fact:** opinions, complaints and personal experiences count as substantive if they meaningfully address the need, especially where the description asks for experiences or responses.
- **Article length:** do not let length decide. A long article that mentions the topic briefly is not relevant. A short article that is entirely about the topic is relevant.

### 6. Borderline cases
Labels are **binary**: there is no "partially relevant". If you are torn, apply the core rule (would a reader with this need get substantive information from this article?), give the label you think is more likely, and set **uncertain = true**. Uncertain cases are reviewed by hand afterwards.

### 7. Evidence
Every label needs:
- a **short reason** (1–2 sentences) explaining the decision in terms of the rule above
- a **supporting excerpt**, copied **word for word** from the article:
  - for a relevant label, the passage that best shows the article addresses the need
  - for a non-relevant label, the passage where the topic comes closest, or `none` if it does not appear at all
