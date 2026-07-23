# Modal Embeddings — 7–9B Panel (SGB-012 / E6 scale test)

Hidden-state embeddings extracted on Modal A100s by `shared/modal_peer_sheaf.py`.
Pulled locally for analysis by `scripts/peer_sheaf_modal_analysis.py` and
`scripts/peer_sheaf_e6_modal_analysis.py`.

## Contents

- `*.npy` — per-model embedding matrices, shape `(n_inputs, hidden_dim)`.
  Filename format: `{ModelID}_{n_inputs}_{seed}.npy`.
- `manifest.json` — run metadata: model list, dataset, n_per_class, seed.

## Panel

| File prefix | Model | hidden_dim |
|-------------|-------|-----------|
| `Yi-1.5-9B` | Yi-1.5-9B-Chat | — |
| `Zephyr-7B` | Zephyr-7B-beta | — |
| `Qwen2.5-7B` | Qwen2.5-7B-Instruct | — |

## Datasets

- `n2000_seed0` files: HH-RLHF chosen/rejected (1000 pairs, seed 0) — used by
  `peer_sheaf_modal_analysis.py` for the E1/E2/E3/E4 pipeline at scale.
- Files without `n2000` prefix are LIARS'-BENCH embeddings per config (convincing-game,
  insider-trading, instructed-deception, alignment-faking variants).

## Reproducibility

To regenerate: `modal run shared/modal_peer_sheaf.py` and then
`modal volume get reward-hacking-results /results/peer_sheaf_e1_modal/ shared/data/cache/peer_sheaf_e1_modal/`.
The `.npy` files are large; re-running is preferred over committing binaries for
configs not already here.
