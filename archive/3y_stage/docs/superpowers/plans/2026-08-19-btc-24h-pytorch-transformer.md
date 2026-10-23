# BTC 24h PyTorch Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train BTC's 24-hour direction-classification Transformer with PyTorch and persist genuine `.pth` checkpoints plus matching `.pkl` inference metadata.

**Architecture:** A focused PyTorch module owns the model, artifact I/O, and inference validation. A dedicated BTC trainer reuses the existing feature and chronological-split utilities to evaluate walk-forward folds and train one deployment checkpoint.

**Tech Stack:** Python, PyTorch, NumPy, pandas, pickle, unittest, existing `lstm_experiment_utils` data pipeline.

**Spec:** `docs/superpowers/specs/2026-08-19-btc-24h-pytorch-transformer-design.md`

## Global Constraints

- Target is BTC's binary `close[t + 24] > close[t]` task only.
- Preserve the 53-feature schema and 24-hour sequence length.
- Save `.pth` files using `torch.save`, never TensorFlow weight conversion.
- Save a matching `.pkl` with feature order, scaler values, sequence length, threshold, target definition, and train coverage.
- Fit scalers on fitting rows and select thresholds from validation rows only.
- Regression, other symbols, and TensorFlow checkpoint conversion are excluded.

---

## File structure

- `src/pytorch_transformer_artifacts.py`: model, artifact persistence, loading, and prediction-schema validation.
- `src/train_btc_24h_pytorch_transformer.py`: walk-forward folds and final-model trainer.
- `tests/test_pytorch_transformer_artifacts.py`: module and artifact contract tests.
- `tests/test_train_btc_24h_pytorch_transformer.py`: synthetic-data smoke training test.
- `requirements-pytorch-transformer.txt`: PyTorch runtime dependency.
- `docs/btc_24h_pytorch_transformer.md`: training and inference instructions.

### Task 1: Implement the model and checkpoint pair

**Files:**

- Create: `src/pytorch_transformer_artifacts.py`
- Create: `tests/test_pytorch_transformer_artifacts.py`
- Create: `requirements-pytorch-transformer.txt`

**Interfaces:**

- Produces `TransformerConfig`, `BtcTransformerClassifier`, `save_artifact_pair`, `load_artifact_pair`, and `predict_latest_probability`.
- Consumes a `(batch, 24, 53)` float tensor and inference metadata.

- [ ] **Step 1: Write the failing forward-shape test.**

```python
def test_classifier_returns_one_logit_for_each_sequence(self):
    model = BtcTransformerClassifier(
        TransformerConfig(feature_count=53, sequence_length=24, d_model=16, ff_dim=32)
    )
    logits = model(torch.zeros((3, 24, 53), dtype=torch.float32))
    self.assertEqual(tuple(logits.shape), (3,))
```

- [ ] **Step 2: Verify red.** Run `python -m unittest tests.test_pytorch_transformer_artifacts.PyTorchTransformerArtifactTests.test_classifier_returns_one_logit_for_each_sequence -v`; expect import failure because the module is absent.

- [ ] **Step 3: Add minimal model code.**

```python
@dataclass(frozen=True)
class TransformerConfig:
    feature_count: int
    sequence_length: int = 24
    d_model: int = 64
    num_heads: int = 4
    ff_dim: int = 128
    dropout: float = 0.20
    num_encoder_layers: int = 1

class BtcTransformerClassifier(nn.Module):
    def forward(self, sequences: torch.Tensor) -> torch.Tensor: ...
```

Use feature projection, learned positions, one `nn.TransformerEncoder`, mean pooling, a 32-unit ReLU head, dropout, and one logit. Reject wrong input shapes.

- [ ] **Step 4: Verify green.** Run the Step 2 command; expect PASS.

- [ ] **Step 5: Write a failing persistence test.**

```python
def test_artifact_pair_round_trips_state_and_metadata(self):
    save_artifact_pair(base, self.model, self.config, self.metadata)
    model, metadata = load_artifact_pair(base.with_suffix('.pth'), base.with_suffix('.pkl'))
    self.assertEqual(metadata['feature_columns'], self.metadata['feature_columns'])
    self.assertTrue(torch.equal(model.state_dict()['head.weight'], self.model.state_dict()['head.weight']))
```

- [ ] **Step 6: Verify red.** Run `python -m unittest tests.test_pytorch_transformer_artifacts.PyTorchTransformerArtifactTests.test_artifact_pair_round_trips_state_and_metadata -v`; expect missing function failure.

- [ ] **Step 7: Implement pair persistence.** `save_artifact_pair` writes `{"format_version": 1, "model_config": asdict(config), "state_dict": model.state_dict()}` using `torch.save` at `<base>.pth` and writes metadata using pickle at `<base>.pkl`. It must require all seven metadata fields in the global constraints. `load_artifact_pair` reconstructs the config and model, calls `load_state_dict`, and sets evaluation mode.

- [ ] **Step 8: Verify green.** Run `python -m unittest tests.test_pytorch_transformer_artifacts -v`; expect all Task 1 tests to pass.

- [ ] **Step 9: Commit.** Run `git add src/pytorch_transformer_artifacts.py tests/test_pytorch_transformer_artifacts.py requirements-pytorch-transformer.txt` and `git commit -m "feat: add PyTorch transformer checkpoint artifacts"`.

### Task 2: Implement strict inference validation

**Files:**

- Modify: `src/pytorch_transformer_artifacts.py`
- Modify: `tests/test_pytorch_transformer_artifacts.py`

**Interfaces:**

- Consumes a loaded model, metadata, and raw-feature pandas DataFrame.
- Produces `predict_latest_probability(model, metadata, rows) -> dict[str, float | bool]`.

- [ ] **Step 1: Write the failing reordered-schema test.**

```python
def test_inference_rejects_reordered_feature_columns(self):
    rows = pd.DataFrame(np.zeros((24, 53)), columns=self.feature_names[::-1])
    with self.assertRaisesRegex(ValueError, 'feature column order'):
        predict_latest_probability(self.model, self.metadata, rows)
```

- [ ] **Step 2: Verify red.** Run `python -m unittest tests.test_pytorch_transformer_artifacts.PyTorchTransformerArtifactTests.test_inference_rejects_reordered_feature_columns -v`; expect failure because prediction validation is absent.

- [ ] **Step 3: Add inference code.** Require an exact metadata column list match and at least 24 finite rows; reject missing, extra, reordered, nonnumeric, or non-finite features. Standardize with stored scaler values, use the latest 24 rows, apply sigmoid to the model logit, and return `{"up_probability": probability, "predict_up": probability >= selected_threshold}`.

- [ ] **Step 4: Verify green.** Run `python -m unittest tests.test_pytorch_transformer_artifacts -v`; expect schema-rejection and valid-probability tests to pass.

- [ ] **Step 5: Commit.** Run `git add src/pytorch_transformer_artifacts.py tests/test_pytorch_transformer_artifacts.py` and `git commit -m "feat: validate PyTorch transformer inference schema"`.

### Task 3: Implement BTC 24h walk-forward and final training

**Files:**

- Create: `src/train_btc_24h_pytorch_transformer.py`
- Create: `tests/test_train_btc_24h_pytorch_transformer.py`

**Interfaces:**

- Consumes BTC CSV through `add_features(..., prediction_horizon=24)`, `GapTimeSeriesSplit`, `split_train_validation`, `standardize_values`, `make_sequences`, and Task 1 artifacts.
- Produces `run_training(...) -> Path`, fold artifact pairs, final artifact pair, metrics CSVs, predictions, history, threshold choices, and manifest.

- [ ] **Step 1: Write a failing one-fold smoke test.**

```python
def test_smoke_training_writes_a_loadable_final_pair(self):
    output_dir = run_training(csv_path=self.btc_csv, output_dir=self.output_dir,
                              folds=1, epochs=1, batch_size=8,
                              test_size=24, inner_valid_size=24)
    model, metadata = load_artifact_pair(
        output_dir / 'checkpoints' / 'btc_24h_final.pth',
        output_dir / 'checkpoints' / 'btc_24h_final.pkl')
    self.assertIn('up_probability', predict_latest_probability(model, metadata, self.feature_rows))
```

The test fixture writes sufficient monotonic hourly OHLCV rows for indicators, a 24-hour target, 24-step sequences, training, validation, and test windows.

- [ ] **Step 2: Verify red.** Run `python -m unittest tests.test_train_btc_24h_pytorch_transformer.Btc24hPyTorchTrainingTests.test_smoke_training_writes_a_loadable_final_pair -v`; expect import failure because the trainer is absent.

- [ ] **Step 3: Implement fold training.** `run_training` must default to `btc_3y.csv`, `folds=5`, `epochs=15`, `batch_size=64`, `test_size=720`, `gap=24`, `max_train_size=5760`, `inner_valid_size=720`, and seed `42`. For each fold, scale only fit rows; train `BCEWithLogitsLoss` with Adam `0.001`, gradient clipping `1.0`, fixed sequence order, and early stopping patience `3`; restore best state. Select the threshold only from validation probabilities using `find_best_threshold`, evaluate the test fold, and write `btc_24h_fold{N}.pth` plus `.pkl`.

- [ ] **Step 4: Implement final training without test-label leakage.** Use the final `inner_valid_size` chronological rows as final validation; fit final scaler on preceding rows, select its threshold from that validation tail, and train the final checkpoint using only the preceding rows with the selected best epoch count. Write `btc_24h_final.pth` and `.pkl`, including fit/validation timestamp coverage.

- [ ] **Step 5: Write outputs.** Create `fold_metrics.csv`, `metrics_summary.csv`, `predictions.csv`, `training_history.csv`, `threshold_choices.csv`, and `manifest.json`. Manifest must record PyTorch version, artifact paths, target, sequence length, 53 features, split settings, and leakage controls.

- [ ] **Step 6: Verify green.** Run the Step 2 command; expect the loadable final pair and inference result.

- [ ] **Step 7: Verify the suite.** Run `python -m unittest discover -s tests -v`; expect legacy and new tests to pass.

- [ ] **Step 8: Commit.** Run `git add src/train_btc_24h_pytorch_transformer.py tests/test_train_btc_24h_pytorch_transformer.py` and `git commit -m "feat: train BTC 24h PyTorch transformer"`.

### Task 4: Document and verify the user workflow

**Files:**

- Create: `docs/btc_24h_pytorch_transformer.md`
- Modify: `README.md`

**Interfaces:**

- Consumes the final command-line trainer and artifact API.
- Produces reproducible training and inference instructions.

- [ ] **Step 1: Write documentation.** Include these commands and an inference example with `load_artifact_pair` and `predict_latest_probability`:

```powershell
python src/train_btc_24h_pytorch_transformer.py --epochs 15 --output-dir models/btc_24h_pytorch_transformer
python src/train_btc_24h_pytorch_transformer.py --smoke-test --output-dir models/btc_24h_pytorch_transformer
```

State that this is a research-direction signal, not financial advice.

- [ ] **Step 2: Run a real-data smoke test.** Run `python src/train_btc_24h_pytorch_transformer.py --smoke-test --output-dir models/btc_24h_pytorch_transformer`; expect exit code 0 and a final pair under `smoke_test/checkpoints/`.

- [ ] **Step 3: Inspect artifacts.** Run `python -c "from pathlib import Path; from src.pytorch_transformer_artifacts import load_artifact_pair; _, d = load_artifact_pair(Path('models/btc_24h_pytorch_transformer/smoke_test/checkpoints/btc_24h_final.pth'), Path('models/btc_24h_pytorch_transformer/smoke_test/checkpoints/btc_24h_final.pkl')); print(d['target_definition'], len(d['feature_columns']))"`; expect the 24-hour target and `53`.

- [ ] **Step 4: Run full verification.** Run `python -m unittest discover -s tests -v`; expect PASS with no regressions.

- [ ] **Step 5: Commit.** Run `git add docs/btc_24h_pytorch_transformer.md README.md` and `git commit -m "docs: explain BTC PyTorch transformer artifacts"`.

## Plan self-review

- Spec coverage: Tasks 1–2 deliver real checkpoints and safe inference; Task 3 preserves the intended leakage controls while producing fold and final artifacts; Task 4 verifies and documents the workflow.
- Placeholder scan: no deferred implementation or vague validation item remains.
- Interface check: Task 3 and Task 4 use the Task 1 interface names consistently.
