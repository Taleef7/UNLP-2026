# Why The Fair 4B Reranker Matters

- Fair 4B full-dev composite: `0.8866`
- Full-dev delta vs `v7`: `+0.0232`
- Bootstrap 95% CI: `+0.0060` to `+0.0414`
- Rerank time: `2130.9179015159607` seconds

The fair 4B result is the right paper comparison because it preserves the `v7` rerank budget and context length. The older 4B regression should stay in the paper only as a methodology warning, not as evidence against larger rerankers.
