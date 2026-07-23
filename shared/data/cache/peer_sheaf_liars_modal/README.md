# Modal Embeddings — LIARS'-BENCH 7–9B Panel (SGB-021 / E6 modal)

Hidden-state embeddings from the 7–9B panel over LIARS'-BENCH configs
(`Cadenza-Labs/liars-bench`), extracted by `shared/modal_peer_sheaf.py`.
These are the primary inputs to the decisive scale test (SGB-020 → SGB-021).

## Contents

- `*.npy` — per-model embeddings for a specific config and seed. Shape `(n_per_class*2, hidden_dim)`.
  Format: `{ModelID}_{config}_{n}_{seed}.npy`.
- `manifest.json` — run metadata.

## Configs present

| Config | Construct | Small panel AUC | 7-9B AUC (length-matched) |
|--------|-----------|-----------------|--------------------------|
| `convincing-game` | Persuasion deception | 0.678 | **0.661** (p=2.5e-6) |
| `insider-trading` | Agentic/strategic | 0.555 (n.s.) | **0.625** (p=8e-8) |
| `instructed-deception` | Overt instructed lie | 0.536 (n.s.) | ~0.54 (n.s.) |
| `alignment-faking-reasoning` | Alignment faking (reasoning) | — | 0.391 (sign-flipped) |
| `alignment-faking-output` | Alignment faking (output) | — | n.s. |
| `gender-secret` | Secret-keeping | — | see `peer_sheaf_modal_summary.json` |

## Reproducibility

To re-embed: `modal run shared/modal_peer_sheaf.py -- --liars` then pull from
the Modal volume. `.npy` files are not committed (binary, large); only text caches
in `peer_sheaf_e6_modal_upload/` are committed.
