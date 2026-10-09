import csv
import json
import numpy as np
from .run import ROOT, OUT, sha
from .inference import load_detector
from ct_lif_study.run import load_data, windows, verify_splits, write_json


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    frozen = json.loads((OUT / 'all_choices_before_test.json').read_text())
    complete = json.loads((OUT / 'complete.json').read_text())
    for relative, expected in protocol['source_hashes'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Frozen source changed: ' + relative)
    for filename, expected in frozen['checkpoint_hashes'].items():
        if sha(OUT / filename) != expected:
            raise ValueError('Checkpoint changed: ' + filename)
    assert complete['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert complete['choices_sha256'] == sha(OUT / 'all_choices_before_test.json')
    assert len(frozen['checkpoint_hashes']) == frozen['fits'] == 288
    with (OUT / 'predictions.csv').open() as f:
        rows = list(csv.DictReader(f))
    data, original = load_data()
    for fold in original['folds']:
        verify_splits(data, fold)
    tested, endpoints = 0, set()
    confusion = {}
    for fold in frozen['folds']:
        raw, y, records = windows(data, fold['splits']['test'])
        expected_keys = [(r['group'], r['end_sample']) for r in records]
        endpoints.update(expected_keys)
        for choice in fold['selected']:
            key = choice['model'], choice['policy']
            saved = [r for r in rows if (r['model'], r['policy'], int(r['held']), int(r['seed'])) == (*key, fold['held'], fold['seed'])]
            assert [(int(r['group']), int(r['end_sample'])) for r in saved] == expected_keys
            np.testing.assert_array_equal(y, [int(r['label']) for r in saved])
            model = load_detector(fold['held'], fold['seed'], choice['policy'], choice['model'])
            result = model.predict(raw.reshape(len(raw), raw.shape[1], 54))
            np.testing.assert_allclose(result['probability'], [float(r['probability']) for r in saved], atol=1e-12, rtol=1e-12)
            np.testing.assert_array_equal(result['decision'], [int(r['decision']) for r in saved])
            p = result['decision']
            counts = [int(((y == label) & (p == prediction)).sum()) for label, prediction in [(1,1),(0,0),(0,1),(1,0)]]
            confusion[key] = confusion.get(key, np.zeros(4, dtype=int)) + counts
            tested += len(saved)
    analysis = json.loads((OUT / 'analysis.json').read_text())
    for key, counts in confusion.items():
        row = next(r for r in analysis['aggregate'] if (r['model'], r['policy']) == key)
        np.testing.assert_array_equal(counts, [row[k] for k in ['tp','tn','fp','fn']])
        assert counts.sum() == 12018
    assert len(endpoints) == 6009 and len({g for g, _ in endpoints}) == 87
    result = {'frozen_sources_and_288_checkpoints_unchanged': True, 'original_splits_match': True,
              'inference_decisions_verified': tested, 'ct_v3_decisions_verified': 24036,
              'unique_endpoints': len(endpoints), 'eligible_trials': 87, 'confusion_counts_match': True}
    write_json(OUT / 'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
