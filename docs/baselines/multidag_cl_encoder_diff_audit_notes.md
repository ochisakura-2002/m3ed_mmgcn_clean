# MultiDAG+CL encoder diff audit notes

## 1. Purpose

Confirm that the MultiDAG+CL encoder refinement did not pollute the MMGCN official train/eval entrypoints.

## 2. Git status before audit

```text
 M configs/smoke/train_multidag_cl_smoke.yaml
 M models/baselines/multidag_cl/__init__.py
 M models/baselines/multidag_cl/multidag_cl_model.py
 M scripts/baselines/debug_multidag_cl_real_batch.py
 M scripts/baselines/train_multidag_cl_smoke.py
 M scripts/evaluate_checkpoint.py
 M scripts/train_mmgcn.py
?? AGENTS.md
?? configs/smoke/train_mmgcn_smoke.yaml
?? datasets/smoke/
?? docs/baselines/baseline_rescreen_report.md
?? docs/baselines/mm_dfn_smoke_review.md
?? docs/baselines/multidag_cl_encoder_audit_notes.md
?? docs/baselines/sdt_isolated_smoke_review.md
?? docs/codex_workflow.md
?? docs/experiment_protocol.md
?? docs/git_workflow.md
?? docs/module_implementation_spec.md
?? docs/project_map.md
?? docs/smoke_test_protocol.md
```

## 3. Unexpected modified files

`scripts/train_mmgcn.py` and `scripts/evaluate_checkpoint.py` were already modified at audit start. Their diffs add `MMGCN_SMOKE` fake dataloader support and do not import or reference MultiDAG+CL.

These files have pre-existing modifications and were not touched further.

## 4. Diff summary for train_mmgcn.py

`scripts/train_mmgcn.py` adds:

- `from datasets.smoke import build_mmgcn_smoke_dataloader`
- an `MMGCN_SMOKE` branch in `build_dataloader(...)`
- an updated unsupported dataset message that lists `MMGCN_SMOKE`
- a final newline change

No MultiDAG+CL model, script, config, train/eval pipeline, or encoder logic is connected to this file by the current diff.

## 5. Diff summary for evaluate_checkpoint.py

`scripts/evaluate_checkpoint.py` adds:

- `from datasets.smoke import build_mmgcn_smoke_dataloader`
- an `MMGCN_SMOKE` branch in `build_dataloader(...)`
- an updated unsupported dataset message that lists `MMGCN_SMOKE`
- a final newline change

No MultiDAG+CL model, script, config, train/eval pipeline, or encoder logic is connected to this file by the current diff.

## 6. Action taken

No revert was performed.

Reason: the MMGCN entrypoint diffs are unrelated to the MultiDAG+CL encoder refinement and match the known pre-existing local MMGCN smoke-test support. Reverting them in this audit would risk discarding earlier smoke work without a clear instruction to do so.

MultiDAG+CL encoder changes were retained in:

- `models/baselines/multidag_cl/__init__.py`
- `models/baselines/multidag_cl/multidag_cl_model.py`
- `scripts/baselines/train_multidag_cl_smoke.py`
- `scripts/baselines/debug_multidag_cl_real_batch.py`
- `configs/smoke/train_multidag_cl_smoke.yaml`

## 7. Git status after audit

```text
 M configs/smoke/train_multidag_cl_smoke.yaml
 M models/baselines/multidag_cl/__init__.py
 M models/baselines/multidag_cl/multidag_cl_model.py
 M scripts/baselines/debug_multidag_cl_real_batch.py
 M scripts/baselines/train_multidag_cl_smoke.py
 M scripts/evaluate_checkpoint.py
 M scripts/train_mmgcn.py
?? AGENTS.md
?? configs/smoke/train_mmgcn_smoke.yaml
?? datasets/smoke/
?? docs/baselines/baseline_rescreen_report.md
?? docs/baselines/mm_dfn_smoke_review.md
?? docs/baselines/multidag_cl_encoder_audit_notes.md
?? docs/baselines/multidag_cl_encoder_diff_audit_notes.md
?? docs/baselines/sdt_isolated_smoke_review.md
?? docs/codex_workflow.md
?? docs/experiment_protocol.md
?? docs/git_workflow.md
?? docs/module_implementation_spec.md
?? docs/project_map.md
?? docs/smoke_test_protocol.md
```

## 8. Smoke test results after audit

`py_compile`: passed after rerunning outside the sandbox with `PYTHONPYCACHEPREFIX`, because direct sandboxed `py_compile` could not atomically replace `.pyc` files.

Compiled files:

- `models/baselines/multidag_cl/__init__.py`
- `models/baselines/multidag_cl/multidag_cl_model.py`
- `scripts/baselines/debug_multidag_cl_forward.py`
- `scripts/baselines/train_multidag_cl_smoke.py`
- `scripts/baselines/debug_multidag_cl_real_batch.py`

Fake forward: passed.

- M3ED fake logits shape: `[2, 4, 7]`
- M3ED fake loss finite: `True`
- M3ED fake backward ok: `True`
- M3ED fake future leakage: `False`
- IEMOCAP fake logits shape: `[2, 5, 6]`
- IEMOCAP fake loss finite: `True`
- IEMOCAP fake backward ok: `True`
- IEMOCAP fake future leakage: `False`

Fake smoke training: passed.

- Command: `conda run -n m3ed_mmgcn python scripts/baselines/train_multidag_cl_smoke.py --config configs/smoke/train_multidag_cl_smoke.yaml`
- Best epoch: `1`
- Best validation Weighted-F1: `0.4190`
- Reload validation Weighted-F1: `0.4190`
- Saved checkpoint: `outputs\smoke\multidag_cl\20260708_105406_multidag_cl_smoke\best_model.pt`
- Saved reload metrics: `outputs\smoke\multidag_cl\20260708_105406_multidag_cl_smoke\smoke_eval_reload_metrics.csv`

## 9. Recommendation for next task

The MultiDAG+CL encoder refinement remains isolated and smoke-passed. It is technically ready for the next MultiDAG+CL train/eval/pipeline integration task, but that task should start only after the pre-existing MMGCN smoke entrypoint diffs are either explicitly accepted or handled separately.

Required answers:

1. `scripts/train_mmgcn.py` is still modified.
2. `scripts/evaluate_checkpoint.py` is still modified.
3. Nothing was reverted.
4. No revert was done because both files contain pre-existing `MMGCN_SMOKE` support unrelated to MultiDAG+CL and were not touched further.
5. MultiDAG+CL encoder changes are still retained.
6. Fake forward still passes.
7. Fake smoke training still passes.
8. Pipeline integration should not be done in this audit turn; it can be the next task after the MMGCN smoke diffs are acknowledged or separated.
