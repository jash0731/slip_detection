import csv
import json
import hashlib
from pathlib import Path
import numpy as np
from .inference import load_detector
from .run import ROOT, OUT
from ct_lif_study.run import load_data, windows, verify_splits


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for relative, expected in protocol['source_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f'Frozen source changed: {relative}')
    completion = json.loads((OUT / 'complete.json').read_text())
    for key, name in [('choices_sha256', 'all_choices_before_test.json'), ('protocol_sha256', 'protocol.json')]:
        if digest(OUT / name) != completion[key]:
            raise ValueError(f'Frozen record changed: {name}')
    frozen = json.loads((OUT / 'all_choices_before_test.json').read_text())
    assert frozen['fits'] == 192 and len(list(OUT.glob('*.npz'))) == 192
    with (OUT / 'predictions.csv').open() as f:
        rows = [row for row in csv.DictReader(f) if row['model'] == 'ct_v2']
    data, original = load_data()
    for fold in original['folds']:
        verify_splits(data, fold)
    tested = 0
    endpoints = set()
    aggregate_counts = {policy: np.zeros(4, dtype=int) for policy in ['balanced', 'fpr20', 'fpr10']}
    for fold in frozen['folds']:
        raw, labels, records = windows(data, fold['splits']['test'])
        expected_keys = [(int(r['group']), int(r['end_sample'])) for r in records]
        endpoints.update(expected_keys)
        for policy in aggregate_counts:
            saved = [r for r in rows if int(r['held']) == fold['held'] and int(r['seed']) == fold['seed'] and r['policy'] == policy]
            observed_keys = [(int(r['group']), int(r['end_sample'])) for r in saved]
            assert observed_keys == expected_keys
            np.testing.assert_array_equal(labels, [int(r['label']) for r in saved])
            detector = load_detector(fold['held'], fold['seed'], policy)
            result = detector.predict(raw.reshape(len(raw), raw.shape[1], 54))
            np.testing.assert_allclose(result['probability'], [float(r['probability']) for r in saved], atol=1e-12, rtol=1e-12)
            np.testing.assert_array_equal(result['decision'], [int(r['decision']) for r in saved])
            decisions = result['decision']
            aggregate_counts[policy] += [(int(((labels == y) & (decisions == p)).sum())) for y, p in [(1, 1), (0, 0), (0, 1), (1, 0)]]
            tested += len(saved)
    analysis = json.loads((OUT / 'analysis.json').read_text())
    for policy, counts in aggregate_counts.items():
        row = next(r for r in analysis['aggregate'] if r['model'] == 'ct_v2' and r['policy'] == policy)
        np.testing.assert_array_equal(counts, [row[k] for k in ['tp', 'tn', 'fp', 'fn']])
        assert counts.sum() == 12018
    assert len(endpoints) == 6009 and len({g for g, _ in endpoints}) == 87
    report = {'frozen_source_and_record_hashes_match': True, 'checkpoints': 192,
              'original_trial_splits_match': True, 'inference_decisions_checked': tested,
              'unique_endpoints': len(endpoints), 'eligible_trials': 87,
              'aggregate_confusion_counts_match': True}
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
