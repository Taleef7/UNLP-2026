# Why Dense Margin Lock Is Neutral-To-Negative

- Full-dev composite: `0.8611`
- Full-dev delta vs `v7`: `-0.0023`
- Doc flip rate: `0.06290672451193059`
- Page change rate: `0.15835140997830802`
- Answer change rate: `0.049891540130151846`

The completed ladder shows exact ties with `v7` on `fold0`, `fold1`, and `lockbox`, followed by a slight regression on `full_dev`. That makes the margin-lock heuristic better interpreted as redundant with the baseline retrieval behavior plus slightly harmful on the broader development mix.
