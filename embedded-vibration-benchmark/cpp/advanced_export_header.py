"""Generate the self-contained Arduino bank from frozen advanced/export/models.json."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def scalar(value):
    if not math.isfinite(value):
        raise ValueError('non-finite export')
    result = format(value, '.9g')
    if '.' not in result and 'e' not in result:
        result += '.0'
    return result + 'f'


def floats(values):
    if isinstance(values[0], list):
        return '{' + ','.join(floats(row) for row in values) + '}'
    return '{' + ','.join(scalar(x) for x in values) + '}'


def integers(values):
    if isinstance(values[0], list):
        return '{' + ','.join(integers(row) for row in values) + '}'
    if any(int(x) != x for x in values):
        raise ValueError('noninteger export')
    return '{' + ','.join(str(int(x)) for x in values) + '}'


def generate(source, destination):
    raw = source.read_bytes()
    data = json.loads(raw)
    assert data['schema'] == 'advanced_tactile_models_v1'
    assert (data['inputs'], data['hidden']) == (6, 16)
    assert len(data['trace']) == data['trace_length'] and all(len(row) == 6 for row in data['trace'])
    input_hash = hashlib.sha256(b''.join(struct.pack('<f', value) for row in data['trace'] for value in row)).hexdigest()
    fixture_id = 'advanced_fixture_' + hashlib.sha256(json.dumps(data['fixture'], sort_keys=True).encode()).hexdigest()[:16]
    lines = ['// Generated from frozen public-data trained models; no hardware measurements.',
             '#pragma once', '#include "advanced_runtime.h"', 'namespace advanced_model {',
             f'constexpr unsigned TRACE_LENGTH={data["trace_length"]};',
             f'constexpr unsigned MODEL_COUNT={len(data["models"])};',
             f'constexpr unsigned SAMPLE_INTERVAL_US={data["sample_interval_us"]};',
             f'constexpr char EXPORT_SHA256[] = "{hashlib.sha256(raw).hexdigest()}";',
             f'constexpr char INPUT_ID[] = "{fixture_id}";',
             f'constexpr char INPUT_HASH[] = "{input_hash}";',
             f'constexpr char INPUT_DESCRIPTION[] = {json.dumps(data["input_description"])};',
             f'constexpr float TRACE[TRACE_LENGTH][advanced_runtime::Inputs] = {floats(data["trace"])};',
             'constexpr advanced_runtime::Model MODELS[MODEL_COUNT] = {']
    for model in data['models']:
        assert len(model['input_weights']) == 16 and all(len(row) == 6 for row in model['input_weights'])
        assert len(model['input_bias']) == len(model['output_weights']) == len(model['output_q']) == 16
        assert len(model['input_q']) == 16 and all(len(row) == 6 for row in model['input_q'])
        assert all(-127 <= x <= 127 for row in model['input_q'] for x in row)
        assert all(-127 <= x <= 127 for x in model['output_q'])
        assert len(model['sha256']) == 64 and all(c in '0123456789abcdef' for c in model['sha256'])
        identity = {key: value for key, value in model.items() if key != 'sha256'}
        assert hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == model['sha256']
        kind = {'lif': 'LIF', 'alif': 'ALIF', 'ann': 'ANN'}[model['kind']]
        fields = [json.dumps(model['model_id']), json.dumps(model['sha256']), 'advanced_runtime::' + kind,
                  floats(model['input_weights']), floats(model['input_bias']), floats(model['output_weights']),
                  scalar(model['output_bias']), scalar(model['decision_logit_threshold']),
                  integers(model['input_q']), integers(model['output_q']), scalar(model['input_scale']),
                  scalar(model['output_scale']), scalar(data['beta']), scalar(data['rho']),
                  scalar(data['base_threshold']), scalar(data['adapt_strength'])]
        lines.append('{' + ','.join(fields) + '},')
    lines.append('};')
    for mode, name in [('float', 'EXPECTED_FLOAT'), ('quant', 'EXPECTED_Q')]:
        lines.append(f'constexpr advanced_runtime::Expected {name}[MODEL_COUNT] = {{')
        for model in data['models']:
            e = model['expected_' + mode]
            assert all(len(e[key]) == 16 for key in ('counts', 'voltage', 'adaptation', 'hidden_sum'))
            fields = [integers(e['counts']), floats(e['voltage']), floats(e['adaptation']),
                      floats(e['hidden_sum']), scalar(e['score'])]
            lines.append('{' + ','.join(fields) + '},')
        lines.append('};')
    lines += ['} // namespace advanced_model', '']
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(lines))
    print(f'Generated {destination}; export SHA256 {hashlib.sha256(raw).hexdigest()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('advanced/export/models.json'))
    parser.add_argument('--destination', type=Path, default=Path('firmware/esp32_advanced_replay/generated_advanced_model.h'))
    args = parser.parse_args()
    generate(args.source, args.destination)
