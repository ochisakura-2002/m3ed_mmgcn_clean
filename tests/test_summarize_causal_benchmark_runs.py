from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from scripts.analyze.summarize_causal_benchmark_runs import (
    write_csv_outputs,
    write_markdown_report,
)


class TestCausalBenchmarkOutputWriting(unittest.TestCase):
    def test_markdown_txt_and_csv_outputs_write_in_temporary_directory(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / "tmp"
        root = temp_root / f"test_causal_benchmark_outputs_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        try:
            output_dir = root / "tables"
            csv_path = output_dir / "summary.csv"
            txt_path = output_dir / "source_files.txt"
            report_path = root / "reports" / "summary.md"
            source_file = Path(__file__)

            write_csv_outputs(
                output_dir,
                {"summary.csv": pd.DataFrame([{"metric": "wf1", "value": 0.5}])},
                [source_file],
            )
            write_markdown_report(
                report_path=report_path,
                review_config_path=source_file,
                selected_runs=pd.DataFrame(columns=["role", "run_id"]),
                completeness=pd.DataFrame(
                    columns=["eligible_for_summary", "run_id", "notes"]
                ),
                protocol=pd.DataFrame(
                    columns=["protocol_consistent", "same_family_config_consistent"]
                ),
                best_validation=pd.DataFrame(),
                test_results=pd.DataFrame(
                    columns=["source_confirmed_as_validation_selected"]
                ),
                statistics=pd.DataFrame(
                    columns=["model", "split", "metric", "mean", "std", "n"]
                ),
                stability=pd.DataFrame(
                    columns=[
                        "model",
                        "val_loss_rebound",
                        "best_val_wf1",
                        "final_val_wf1",
                    ]
                ),
                duplicate=pd.DataFrame(),
                per_class_messages=[],
                source_files=[source_file],
                generated_at="2026-07-13T00:00:00+00:00",
            )

            csv_frame = pd.read_csv(csv_path, encoding="utf-8-sig")
            self.assertEqual(csv_frame.to_dict("records"), [{"metric": "wf1", "value": 0.5}])

            txt_bytes = txt_path.read_bytes()
            self.assertTrue(txt_bytes.endswith(b"\n"))
            self.assertNotIn(b"\r\n", txt_bytes)

            markdown_bytes = report_path.read_bytes()
            self.assertIn(b"# Causal benchmark eight-run review\n", markdown_bytes)
            self.assertTrue(markdown_bytes.endswith(b"\n"))
            self.assertNotIn(b"\r\n", markdown_bytes)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
