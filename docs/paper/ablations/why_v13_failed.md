# Why v13 Failed Despite Positive Local Splits

This note compares the exact `v13` submission family against `v7` across local full-dev and final Kaggle results.

- Local full-dev delta vs `v7`: `+0.0003`
- Kaggle public delta vs `v7`: `-0.0457`
- Kaggle private delta vs `v7`: `-0.0281`
- Full-dev bootstrap 95% CI: `-0.0141` to `+0.0150`

## Risk Signals

- Doc flip rate: `0.06290672451193059`
- Page change rate: `0.1648590021691974`
- Answer change rate: `0.0824295010845987`
- Per-domain score delta: `{"domain_1": -0.015546285954214594, "domain_2": 0.012276801269755867}`

## Interpretation

The `v5_refocus` prompt change appears to improve the local development distribution, but that signal does not survive the public or private competition distributions. The paper should treat this as a prompt-locality effect rather than a robust task improvement.
