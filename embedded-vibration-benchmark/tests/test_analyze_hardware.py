"""Synthetic parser fixtures only: these tests do not produce hardware evidence."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_hardware as analysis


def synthetic_rows():
    rows = []
    for method, interval in analysis.METHODS:
        for width in analysis.WIDTHS:
            for phase in analysis.PHASES:
                trial = len(rows) + 1
                epoch = trial * 200000
                rise = epoch + 40000 + phase
                row = dict.fromkeys(analysis.COLUMNS, "")
                row.update(
                    run_id="hardware-SYNTHETIC-TEST-ONLY", trial_id=str(trial),
                    trial_kind="target", acquisition=method, detector="immediate",
                    poll_interval_ms=interval, core_config="generator=0;acquisition=1;decision=1",
                    load="none;wifi=off;bluetooth=off",
                    parameters=f'width_us={width};phase_us={phase};epoch_us={epoch};'
                               f'requested_rise_us={rise};deadline_us={rise + width + 40000}',
                    pulse_schedule_us=f"[[{rise},{rise + width}]]", first_pulse_us=str(rise),
                    decision_us=str(rise + 12), latency_us="12", outcome="detected",
                    notes='source=hardware;stimulus=gpio_loopback;skipped_polls=0;'
                          'stimulus_after_deadline=0;fixture="SYNTHETIC, NOT MEASURED"',
                )
                rows.append(row)
    return rows


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic.csv"

    def write(self, rows):
        with self.path.open("w", newline="") as handle:
            handle.write("# synthetic test fixture, never hardware evidence\n")
            writer = csv.DictWriter(handle, fieldnames=analysis.COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            handle.write("# COMPLETE: synthetic fixture\n")

    def test_csv_escaping_misses_and_statistics(self):
        rows = synthetic_rows()
        for index, row in enumerate(rows[:100]):
            if index < 20:
                row.update(outcome="miss", decision_us="", latency_us="")
            else:
                row.update(decision_us=str(int(row["first_pulse_us"]) + index), latency_us=str(index))
        self.write(rows)
        parsed = analysis.read_capture(self.path)
        self.assertEqual(parsed[0]["notes"], rows[0]["notes"])
        stats = analysis.summarize(parsed)[0]
        self.assertEqual((stats["trials"], stats["detected"], stats["misses"]), (100, 80, 20))
        self.assertEqual(stats["miss_rate"], 0.2)
        self.assertEqual((stats["median_us"], stats["p95_us"], stats["max_us"]), (59.5, 95, 99))

    def test_all_misses_have_no_latency_statistics(self):
        rows = synthetic_rows()
        for row in rows:
            row.update(outcome="miss", decision_us="", latency_us="")
        self.write(rows)
        stats = analysis.summarize(analysis.read_capture(self.path))[0]
        self.assertEqual(stats["miss_rate"], 1)
        self.assertEqual((stats["median_us"], stats["p95_us"], stats["max_us"]), ("", "", ""))

    def test_false_alarm_and_timing_flags_retained(self):
        rows = synthetic_rows()
        rows[0].update(outcome="false_alarm", decision_us=str(int(rows[0]["first_pulse_us"]) - 10), latency_us="-10")
        rows[1]["notes"] += ";skipped_polls=2"
        self.write(rows)
        stats = analysis.summarize(analysis.read_capture(self.path))[0]
        self.assertEqual((stats["detected"], stats["pre_stimulus_alarms"], stats["timing_flagged_trials"]), (99, 1, 1))

    def test_incomplete_duplicate_mixed_and_corrupt_captures_rejected(self):
        for mutation, expected in (
            (lambda r: r.pop(), "incomplete"),
            (lambda r: r[1].update(trial_id="1"), "duplicate"),
            (lambda r: r[1].update(run_id="hardware-OTHER"), "one run_id"),
            (lambda r: r[0].update(latency_us="99"), "inconsistent"),
            (lambda r: r[0].update(outcome="miss"), "empty decision"),
            (lambda r: r[0].update(pulse_schedule_us="[]"), "pulse_schedule_us"),
            (lambda r: r[0].update(parameters=r[1]["parameters"]), "actual pulse"),
        ):
            with self.subTest(expected=expected):
                rows = synthetic_rows()
                mutation(rows)
                self.write(rows)
                with self.assertRaisesRegex(ValueError, expected):
                    analysis.read_capture(self.path)

    def test_simulation_and_unmarked_rows_rejected(self):
        self.path.write_text("experiment,trial,kind,method\nacquisition,0,target,ideal\n")
        with self.assertRaisesRegex(ValueError, "simulation CSV"):
            analysis.read_capture(self.path)
        rows = synthetic_rows()
        rows[0]["notes"] = "source=simulated;stimulus=gpio_loopback"
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "not labeled"):
            analysis.read_capture(self.path)


if __name__ == "__main__":
    unittest.main()
