"""Recorded timestamp support spans; these are not measured detector latency."""
import json
from pathlib import Path

import numpy as np

from .common import ROOT, read_data
from .analyze import read_csv
from .experiments import OUT, write_csv


def run():
    data, _, _ = read_data()
    predictions = read_csv(OUT / 'selected_predictions.csv')
    endpoints = sorted({(int(r['group']), int(r['end_sample'])) for r in predictions})
    groups = {int(group): np.flatnonzero(data['group'] == group) for group in np.unique(data['group'])}
    rows = []
    for group, end in endpoints:
        indices = groups[group]
        times = data['t_ms'][indices]
        for window in (16, 32, 64):
            rows.append(dict(source='public_recorded_timestamp_audit', group=group, end_sample=end,
                             object=str(data['object_names'][data['object'][indices[end]]]), window=window,
                             frame_span_ms=float(times[end] - times[end - window + 1]),
                             derivative_raw_support_ms=float(times[end] - times[end - window]) if end >= window else None))
    write_csv(OUT / 'observation_spans.csv', rows)
    summary = []
    for obj in data['object_names']:
        for window in (16, 32, 64):
            selected = [r for r in rows if r['object'] == obj and r['window'] == window]
            values = np.array([r['derivative_raw_support_ms'] for r in selected if r['derivative_raw_support_ms'] is not None])
            summary.append(dict(source='public_recorded_timestamp_audit', object=str(obj), window=window,
                                eligible_endpoints=len(selected), raw_support_median_ms=float(np.median(values)),
                                raw_support_p95_ms=float(np.percentile(values, 95)), raw_support_min_ms=float(values.min()),
                                raw_support_max_ms=float(values.max())))
    write_csv(OUT / 'observation_summary.csv', summary)
    print('Recorded timestamp support spans saved; no physical detection delay or ESP32 measurement.')


if __name__ == '__main__':
    run()
