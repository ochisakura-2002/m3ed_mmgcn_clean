# Canonical experiment output layout

All future experiment writes use one date-centered, group-scoped hierarchy:

```text
outputs/
  <YYYYMMDD>/
    <experiment_group>/
      runs/
        <run_id>/
          checkpoints/
          logs/
          metrics/
          artifacts/
          resolved_config.yaml
      logs/
        launcher/
          <batch_id>/
      manifests/
        batches/
          <batch_id>/
            resolved_configs/
            commands.txt
            matrix.yaml
            matrix.csv
            git_commit.txt
            preparation_metadata.json
      review/
        batches/
          <batch_id>/
      reports/
        batches/
          <batch_id>/
      analysis/
        batches/
          <batch_id>/
```

`outputs/<YYYYMMDD>/<experiment_group>` is the experiment root. The only
third-level functional directory names are `runs`, `logs`, `manifests`,
`review`, `reports`, and `analysis`.

## Identifiers

- `experiment_date` is the launcher or single-entry start date on the runtime
  machine, formatted as a valid local-calendar `YYYYMMDD`. One pipeline freezes
  this value before launching children, so crossing midnight does not split a
  batch.
- `experiment_group` is a stable, readable experiment-family identifier. It
  must not contain a date, path separator, or per-run suffix. Runs from the
  same matrix or config family share one group. Examples are
  `formal_long32_primary_seed42`, `formal_long32_multiseed`,
  `causal8_original8_formal`, and `mmgcn_modality_ablation`.
- `run_id` identifies one training run. It may encode model, context, split,
  seed, timestamp, or a collision-resistant suffix, but it is never used as
  the experiment group.
- `batch_id` identifies one launcher/preparation attempt. Launcher logs and
  batch-owned manifest, review, report, and analysis artifacts are isolated by
  this identifier.

The common resolver is `utils/output_paths.py`. Resolution priority for
`experiment_group` is explicit CLI, config, canonical config-family path, then
an explicit stable caller default. Output-producing entrypoints must use this
resolver instead of hand-building paths. Automatically inferred groups are
deterministically bounded for Windows path safety; an eight-character content
hash preserves the identity of a longer readable config-family prefix.

## Repeats, resume, and collision policy

Generated single-run IDs use an atomic directory creation with a time and
random suffix. A matrix-owned fixed `run_id` also uses atomic
`mkdir(exist_ok=False)`: if the canonical run directory already exists,
preparation or training fails before the first epoch. Existing output is never
deleted, moved, overwritten, or silently reused.

Repeated launch/preparation of the same date and group must use a new
`batch_id`. Reusing an existing batch directory fails rather than merging
manifests. Resume is the only intentional reuse path and continues in the
checkpoint's original run directory, including for a legacy run.

## Long-training groups

The primary and optional matrices use:

```text
formal_long32_primary_seed42
formal_long32_multiseed
```

Their resolved configs and preparation metadata live under
`manifests/batches/<batch_id>/`; each resolved config's `output.root` is the
exact corresponding canonical `runs/<run_id>` directory.

## Legacy read compatibility

The path helpers and analysis discovery retain read-only support for:

```text
outputs/<YYYYMMDD>/runs/
outputs/<YYYYMMDD>/analysis/
outputs/runs/
outputs/analysis/
outputs/long_training/primary/
outputs/long_training/multi_seed/
outputs/launcher_logs/
```

These locations are not valid targets for new writes. Historical outputs are
not migrated, renamed, or deleted.

The formal remote 32-run batch reported as currently running was launched from
the old layout at commit `d8a7701` or the remote machine's actual
`git rev-parse HEAD`. It remains untouched: do not stop it, modify its
workspace, request `git pull`, or move its output. This policy applies only to
future locally prepared code and future batches after normal Git handoff.
