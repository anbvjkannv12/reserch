# BTC 24h PyTorch Transformer Checkpoint Design

## Goal

Train a BTC 24-hour direction-classification Transformer in PyTorch and save
artifacts that can be loaded for later inference without TensorFlow.

## Scope

- Keep the existing hourly BTC feature pipeline, 53-feature schema, 24-step
  input sequence, and chronological walk-forward validation approach.
- Train the same Transformer Encoder family in PyTorch for the 24-hour binary
  classification target: `close[t + 24] > close[t]`.
- Write a PyTorch checkpoint for every validation fold and one final model
  trained on all eligible historical rows.
- Write a matching pickle metadata artifact for every checkpoint.

Regression, other assets, and conversion of legacy TensorFlow weights are out
of scope.

## Artifacts

Artifacts are written beneath `models/btc_24h_pytorch_transformer/`:

- `checkpoints/btc_24h_fold{N}.pth`: a `torch.save` dictionary containing the
  model `state_dict`, architecture configuration, and checkpoint identity.
- `checkpoints/btc_24h_fold{N}.pkl`: inference metadata containing feature
  column order, feature-scaler mean and standard deviation, sequence length,
  selected classification threshold, target definition, and training-data
  coverage.
- `checkpoints/btc_24h_final.pth` and `checkpoints/btc_24h_final.pkl`: the
  deployment-oriented final pair, trained after validation selection using all
  eligible rows.
- Standard CSV metrics, predictions, histories, and a JSON manifest documenting
  the run configuration and artifact paths.

The `.pth` files are genuine PyTorch checkpoints and are loadable through
`torch.load(..., weights_only=True)` for the stored tensor state. The `.pkl`
files intentionally contain no model weights.

## Model and training

The PyTorch module accepts `(batch, sequence_length, feature_count)` tensors.
It projects features to `d_model`, adds learned positional embeddings, applies
one `TransformerEncoderLayer`, performs mean temporal pooling, and predicts one
upward-move logit. Training uses `BCEWithLogitsLoss`, Adam with gradient-norm
clipping, no data shuffling, and validation-loss early stopping with best
weights restored.

For each chronological fold, the feature scaler is fit only on the fold's
training indexes. Threshold selection runs only on its validation window. The
final model fits its scaler and threshold only on rows preceding its held-out
final-validation window; it does not use any test-fold labels to choose a
threshold.

## Inference contract

The loader accepts raw feature rows in exactly the stored feature-column order,
applies the stored scaler, keeps the latest 24 rows, and returns a probability
plus the stored thresholded direction. It rejects missing, extra, or reordered
features before inference.

## Verification

Tests cover checkpoint round-tripping, metadata round-tripping, forward-pass
shape, and feature-schema validation. A one-fold, one-epoch smoke run verifies
that the script writes a matching `.pth`/`.pkl` pair and that the loader can
produce an inference from the saved final artifact.
