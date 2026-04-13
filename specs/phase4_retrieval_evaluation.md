# Phase 4 Spec: Retrieval Evaluation

## Goal

Add retrieval-based validation metrics that measure whether a predicted EEG embedding retrieves the correct phrase from a candidate bank better than competing phrases.

## Inputs

- `pred_embeddings`: float array with shape `(n_samples, embed_dim)`
- Candidate phrase bank:
  - `bank_embeddings`: float array with shape `(n_bank, embed_dim)`
  - optional `bank_phrases`: sequence of `n_bank` phrase strings
- Ground-truth identifiers for each prediction:
  - either `target_indices`: int array with shape `(n_samples,)`, or
  - `target_phrases`: sequence of `n_samples` phrase strings plus `bank_phrases`

## Outputs

- Retrieval metrics dict with:
  - `top1`
  - `top5`
  - `top10`
  - `mrr`
- Optional `target_count` and `bank_size` metadata for logging/debugging

## Processing

1. Validate that predicted embeddings and bank embeddings are non-empty 2D arrays with matching embedding dimension.
2. L2-normalize both arrays.
3. Compute cosine similarity matrix `(n_samples, n_bank)`.
4. Resolve each sample's correct bank row via `target_indices` or `target_phrases`.
5. Rank bank candidates per sample by descending cosine similarity.
6. Compute top-k hit rate and reciprocal rank.

## Edge Cases

- Empty arrays: raise `ValueError` with a clear message.
- Mismatched embedding dimensions: raise `ValueError`.
- `k > bank_size`: clip `k` to `bank_size`.
- Missing target phrase in bank: raise `KeyError`.
- Missing ground-truth bank mapping entirely: raise `ValueError` explaining the minimum required input.

## Success Criteria

- Retrieval metrics can be computed from arrays without new heavy dependencies.
- The training pipeline can optionally log retrieval metrics while keeping cosine evaluation available for compatibility.
- Unit tests cover happy path, phrase-based target resolution, and key failure cases.
