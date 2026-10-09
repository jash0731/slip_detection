"""Fabricated capture fixtures exercise validation, not hardware performance."""

import csv
import io
import unittest

from analyze_replay import FIELDS, summarize, validate_capture


def fixture(pacing="throughput"):
    rows = []
    for mode in ("dense", "sparse"):
        for step in range(2):
            rows.append(dict(zip(FIELDS, ["hardware_replay", "fixture-only", "test-model", "test-input", 1, mode, pacing, step, 1, 2, 1, 2, 3, 8 if pacing == "paced" else 0, int(pacing == "paced"), 0.25, 1])))
    return rows


def capture(rows):
    stream = io.StringIO()
    stream.write("# trials=1 samples_per_trace=2 inputs=4 hidden=12\n# nominal_interval_us=10\n")
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    stream.write("# COMPLETE. Fixture only, not a hardware run.\n")
    return stream.getvalue()


class ReplayValidationTests(unittest.TestCase):
    def test_complete_pairs_and_paced_misses(self):
        for pacing in ("throughput", "paced"):
            rows, dimensions = validate_capture(capture(fixture(pacing)))
            self.assertEqual(dimensions["samples_per_trace"], 2)
            self.assertEqual(summarize(rows)[0]["deadline_misses"], 2 if pacing == "paced" else 0)

    def test_truncation_and_missing_trial(self):
        for text in (capture(fixture()).split("# COMPLETE.")[0], capture(fixture()[:-1]), capture(fixture()).replace("trials=1", "trials=2")):
            with self.assertRaises(ValueError):
                validate_capture(text)

    def test_source_mixed_identity_duplicates_and_parity(self):
        for field, value in (("source", "synthetic_host"), ("run_id", "different"), ("hidden_spikes", 3), ("final_score", 0.9)):
            rows = fixture()
            rows[-1][field] = value
            with self.assertRaises(ValueError):
                validate_capture(capture(rows))
        with self.assertRaises(ValueError):
            validate_capture(capture(fixture() + [fixture()[0]]))

    def test_invalid_timing_numeric_and_decision(self):
        for field, value in (("duration_us", 4), ("encode_us", -1), ("deadline_miss", 1), ("final_score", "nan"), ("final_decision", 0), ("active_inputs", 5)):
            rows = fixture()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                validate_capture(capture(rows))


if __name__ == "__main__":
    unittest.main()
