"""Fake fixtures test capture integrity only; none are physical measurements."""

import csv
import io
import unittest
from analyze_advanced import FIELDS, summarize, validate_capture


def fixture(drift=False):
    rows = []
    for precision in ("float32", "int8_weights_float_state"):
        changed = drift and precision.startswith("int8")
        for step in range(2):
            rows.append(dict(zip(FIELDS, ["hardware_advanced_replay", "fixture", "model", "a" * 64, "input", "b" * 64, 1, precision, step, 1, 2, 3, 2, -0.25 if changed else 0.25, 0, int(not changed), 1, int(changed)])))
    return rows


def capture(rows):
    stream = io.StringIO()
    stream.write("# trials=1 samples_per_trace=2 models=1 precisions=2 hidden=16\n# fixture_parity=pass fixture_only\n")
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    stream.write("# COMPLETE hardware_advanced_replay fixture_only\n")
    return stream.getvalue()


class AdvancedCaptureTests(unittest.TestCase):
    def test_pairs_and_quantized_drift_are_valid(self):
        rows, dimensions = validate_capture(capture(fixture(True)))
        self.assertEqual(dimensions["models"], 1)
        self.assertEqual(summarize(rows)[1]["drifted_traces"], 1)

    def test_truncated_duplicate_and_wrong_source(self):
        for text in (capture(fixture()).split("# COMPLETE")[0], capture(fixture()[:-1]), capture(fixture() + [fixture()[0]]), capture(fixture()).replace("hardware_advanced_replay,", "public_recorded_host,")):
            with self.assertRaises(ValueError):
                validate_capture(text)

    def test_hash_timing_and_flag_integrity(self):
        for key, value in (("model_hash", "bad"), ("input_hash", "c" * 64), ("run_id", "other"), ("duration_us", 4), ("final_decision", 0), ("decision_drift", 1), ("final_score", "nan")):
            rows = fixture()
            rows[0][key] = value
            with self.assertRaises(ValueError):
                validate_capture(capture(rows))

    def test_precision_pairs_share_threshold(self):
        rows = fixture()
        for row in rows[2:]:
            row["decision_threshold"] = 0.1
        with self.assertRaises(ValueError):
            validate_capture(capture(rows))


if __name__ == "__main__":
    unittest.main()
