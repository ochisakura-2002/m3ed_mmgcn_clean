# Legacy Model Imports After Script Phase 3B2

## Audit scope and method

The audit parsed every Git-tracked Python file under `scripts/` and `tests/`,
plus the Phase 3B2 wrappers, package files, and compatibility test that are
present in the working tree but not yet staged. The two protected local
group-meeting files were excluded and were not parsed:

- `scripts/analyze/export_group_meeting_baseline_report.py`
- `tests/analyze/test_group_meeting_baseline_report.py`

Python `ast` counted one dependency edge for each `import` or `from` statement
whose module is `models.baselines` or one of its descendants. The independent
text cross-check used:

```text
git grep -n -E "models\.baselines|original_repro|causal_baseline_registry" -- scripts tests
git grep -n -E "^[[:space:]]*(from|import)[[:space:]]+models\.baselines" -- scripts tests
```

All 188 audited Python files parsed successfully. The broader grep also found
runtime protocol names such as `original_repro_*`; those are compatibility
identifiers rather than Python imports and were not renamed.

## Results

| Scope | Legacy model import edges | Status |
|---|---:|---|
| Canonical execution | 0 | PASS |
| Canonical analysis | 0 | PASS |
| Canonical support | 0 | PASS |
| Compatibility wrappers | 0 | PASS |
| Deferred production scripts | 0 | PASS |
| Tests | 10 | Dedicated compatibility coverage only |

- Canonical support Python files audited: 48.
- Support compatibility wrappers audited: 29.
- All production script compatibility wrappers: 63.
- Canonical support legacy model imports: 0.
- Tracked production script legacy model imports: 0.
- Remaining legacy imports: 10 edges in one file.

Every remaining edge belongs to
`tests/test_model_layout_compatibility.py`. That test intentionally imports
both old and canonical model paths to prove object identity and strict
state-dict compatibility. The 17 legacy model imports that previously existed
in ordinary behavior tests were changed to canonical model packages without
changing test logic.

## Canonical replacements used in Phase 3B2

- `models.baselines.causal_baseline_registry` →
  `models.registry.causal`
- `models.baselines.dialoguegcn` →
  `models.dialoguegcn.unified`
- `models.baselines.gsmcc` →
  `models.gsmcc.project_variant.causal`
- `models.baselines.mmgcn.mm_gcn` →
  `models.mmgcn.unified.mm_gcn`
- `models.baselines.multidag_cl` →
  `models.multidag_cl.unified`
- `models.baselines.sdt` →
  `models.experimental.sdt`
- `models.baselines.simple_mlp` →
  `models.simple_mlp.model`
- `models.baselines.original_repro` →
  `models.registry.paper_aligned`
- `models.baselines.original_repro.gsmcc.model` →
  `models.gsmcc.project_variant.full_context.model`

The support implementations that previously imported old execution runtime
wrappers now import `scripts.runtime.causal_graph` or
`scripts.runtime.paper_aligned` directly.

## Removal gate

The ten test-only edges remain until the model compatibility wrappers and
`tests/test_model_layout_compatibility.py` are retired together in a separate
wrapper-removal task. Phase 3B2 does not delete any compatibility wrapper.
