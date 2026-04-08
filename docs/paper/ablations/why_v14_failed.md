# Why v14 Failed Despite Strong Local Offline Results

This note compares the dense-doc-lock-v3 family against `v7` across local full-dev and final Kaggle results.

- Local full-dev delta vs `v7`: `+0.0178`
- Kaggle public delta vs `v7`: `-0.0368`
- Kaggle private delta vs `v7`: `-0.0326`
- Full-dev bootstrap 95% CI: `+0.0042` to `+0.0315`

## Risk Signals

- Doc flip rate: `0.06941431670281996`
- Page change rate: `0.16702819956616052`
- Answer change rate: `0.049891540130151846`
- Per-domain score delta: `{"domain_1": 0.03509691617137292, "domain_2": 0.004735609888260304}`

## Interpretation

The dense-doc-lock heuristic improves the local development split while failing on public and private Kaggle. That pattern is consistent with overfitting to long-document proxies and with domain mismatch between local development and the hidden competition distribution.
