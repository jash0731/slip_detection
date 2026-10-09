"""Artificial capture fixtures verify parsing; their timings are not measurements."""

import csv
import io
import unittest
from analyze_activity import FIELDS, summarize, validate_capture


def fixture():
    rows = []
    for active in range(3):
        for mode in ("dense", "sparse"):
            rows.append(dict(zip(FIELDS, ["hardware_activity_sweep", "fixture-only", "test-model", 1, mode, active, 2, 10, 20 if mode == "dense" else 15, 60 if mode == "dense" else active * 30, 30, active * 0.5])))
    return rows


def capture(rows):
    stream = io.StringIO()
    stream.write("# trials=1 steps=10 conditions=3 inputs=2 hidden=3\n")
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    stream.write("# activity_state_parity=pass fixture_only\n# COMPLETE hardware_activity_sweep fixture_only\n")
    return stream.getvalue()


class ActivityValidationTests(unittest.TestCase):
    def test_complete_pairs_and_batch_mean_statistics(self):
        rows, dimensions = validate_capture(capture(fixture()))
        self.assertEqual(dimensions["conditions"], 3)
        self.assertEqual(summarize(rows)[0]["sparse_dense_ratio"], 0.75)

    def test_truncation_duplicates_and_missing_conditions(self):
        for text in (capture(fixture()).split("# COMPLETE")[0], capture(fixture()[:-1]), capture(fixture() + [fixture()[0]]), capture(fixture()).replace("activity_state_parity=pass", "activity_state_parity=fail")):
            with self.assertRaises(ValueError):
                validate_capture(text)

    def test_source_dimensions_counts_and_identity(self):
        for field, value in (("source", "host_reference"), ("run_id", "other"), ("steps", 11), ("hidden_updates", 29), ("input_contributions", 59), ("duration_us", 0), ("checksum", "nan")):
            rows = fixture()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                validate_capture(capture(rows))

    def test_pair_checksum_mismatch(self):
        rows = fixture()
        rows[-1]["checksum"] = 99
        with self.assertRaises(ValueError):
            validate_capture(capture(rows))


if __name__ == "__main__":
    unittest.main()
