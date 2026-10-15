import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .model import Config, Kernel, Preprocessor, config_dict, merge_events, score_positive
from .replay import replay


def pattern(length, acceleration_ticks=(), increase_ticks=(), decrease_ticks=(), axes=1):
    return [([int(i in acceleration_ticks and j < axes) for j in range(3)],
             i in increase_ticks, i in decrease_ticks, True) for i in range(length)]


def trace(inputs, config=Config()):
    baseline, proposed = Kernel(config, False), Kernel(config, True)
    b, p, extra = [], [], []
    for i, values in enumerate(inputs):
        rb, rp = baseline.step(*values), proposed.step(*values)
        if rb['spike']:
            b.append(i)
        if rp['spike']:
            p.append(i)
        extra.append(rp)
    return dict(baseline_spike_ticks=b, proposed_spike_ticks=p,
                max_inhibition_500ms=max(r['inhibition_500ms'] for r in extra),
                max_excitation_500ms=max(r['excitation_500ms'] for r in extra)), extra


def validate():
    checks = []
    cases = {}
    cases['pressure_change_without_acceleration'], extra = trace(
        pattern(1000, increase_ticks=range(0, 500), decrease_ticks=range(500, 1000)))
    assert cases['pressure_change_without_acceleration']['proposed_spike_ticks'] == []
    assert all(r['excitation'] == 0 for r in extra)
    checks.append('pressure decrease cannot excite the network without preceding acceleration')
    cases['constant_pressure_acceleration_burst'], _ = trace(pattern(20, acceleration_ticks=[0, 3, 6, 9]))
    assert cases['constant_pressure_acceleration_burst']['baseline_spike_ticks']
    assert (cases['constant_pressure_acceleration_burst']['baseline_spike_ticks']
            == cases['constant_pressure_acceleration_burst']['proposed_spike_ticks'])
    checks.append('acceleration alone reaches threshold with constant pressure')
    cases['pressure_increase_can_suppress_weak_evidence'], _ = trace(
        pattern(20, acceleration_ticks=[0, 3, 6, 9], increase_ticks=[0, 5]))
    assert cases['pressure_increase_can_suppress_weak_evidence']['baseline_spike_ticks']
    assert not cases['pressure_increase_can_suppress_weak_evidence']['proposed_spike_ticks']
    checks.append('counterexample: inhibition cap does not guarantee weak-evidence detection')
    cases['pressure_decrease_can_create_alarm_after_acceleration'], _ = trace(
        pattern(20, acceleration_ticks=[0, 2, 4], decrease_ticks=[4]))
    assert not cases['pressure_decrease_can_create_alarm_after_acceleration']['baseline_spike_ticks']
    assert cases['pressure_decrease_can_create_alarm_after_acceleration']['proposed_spike_ticks']
    checks.append('counterexample: recent acceleration plus pressure decrease can create an alarm')
    cases['isolated_three_axis_impact'], _ = trace(pattern(20, acceleration_ticks=[0], axes=3))
    assert cases['isolated_three_axis_impact']['baseline_spike_ticks'] == [0]
    assert cases['isolated_three_axis_impact']['proposed_spike_ticks'] == [0]
    checks.append('isolated impact is not automatically rejected')
    kernel = Kernel()
    first = kernel.step([True, False, False], False, True, True)
    assert first['excitation'] == 0
    for _ in range(21):
        kernel.step([False] * 3, False, False, True)
    assert kernel.step([False] * 3, False, True, True)['excitation'] == 0
    kernel.step([False] * 3, False, False, False)
    assert kernel.last_acceleration is None and kernel.output.voltage == 0
    assert kernel.step([False] * 3, False, True, True)['excitation'] == 0
    checks.append('current acceleration is excluded from the preceding-100-ms guard and contact loss resets state')
    rng = np.random.default_rng(20261015)
    random_inputs = []
    for i in range(20000):
        axes = (rng.random(3) < .18).tolist()
        random_inputs.append((axes, bool(rng.random() < .12), bool(rng.random() < .12), i % 3000 != 0))
    baseline, proposed = Kernel(pressure_enabled=False), Kernel()
    ablated = Kernel(replace(Config(), pressure_increase_weight=0., pressure_decrease_weight=0.))
    expected = []
    for values in random_inputs:
        b, p, a = baseline.step(*values), proposed.step(*values), ablated.step(*values)
        assert b['spike'] == a['spike']
        assert 0 <= p['inhibition_500ms'] <= .25 + 1e-12
        assert 0 <= p['excitation_500ms'] <= .25 + 1e-12
        expected.append([int(b['spike']), int(p['spike']), int(p['acceleration_fired']),
                         p['inhibition'], p['excitation'], p['inhibition_500ms'], p['excitation_500ms']])
    checks.append('20,000 randomized updates obey both budgets and zero-pressure ablation matches baseline')
    cases['repeated_pressure_budget_stress'], _ = trace(
        pattern(1000, acceleration_ticks=range(0, 1000, 2),
                increase_ticks=range(0, 1000, 5), decrease_ticks=range(1, 1000, 5)))
    assert cases['repeated_pressure_budget_stress']['max_inhibition_500ms'] <= .25 + 1e-12
    assert cases['repeated_pressure_budget_stress']['max_excitation_500ms'] <= .25 + 1e-12
    pre = Preprocessor()
    for _ in range(500):
        events = pre.step([0., 0., 1.], 1500.)
        assert not any(events[0]) and not events[1] and not events[2]
    assert events[3]
    for _ in range(200):
        events = pre.step([0., 0., 1.], 100.)
    assert not events[3]
    checks.append('constant gravity and contact initialization produce no acceleration or pressure-change spikes')
    assert merge_events([0., 100., 300., 600.]) == [0., 600.]
    early = score_positive([2900.], 3000., 3500., 60.)
    assert early['early_events'] == 1 and not early['upper_bound_detected']
    ambiguous = score_positive([3000.], 3000., 3500., 60.)
    assert ambiguous['upper_bound_detected'] and not ambiguous['conservative_detected']
    detected = score_positive([2900., 3050.], 3000., 3500., 60.)
    assert detected['conservative_detected'] and detected['delay_ms'] == 50.
    checks.append('event merging, early-alarm rejection and camera-frame ambiguity are handled explicitly')
    with tempfile.TemporaryDirectory(prefix='pressure_lif_validation_') as directory:
        binary = Path(directory) / 'host_replay'
        source = Path(__file__).resolve().parent / 'host_replay.cpp'
        subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', str(source), '-o', str(binary)], check=True)
        text = ''.join(' '.join(str(int(v)) for v in [*a, up, down, contact]) + '\n'
                       for a, up, down, contact in random_inputs)
        cpp = subprocess.run([str(binary)], input=text, text=True, capture_output=True, check=True)
        actual = np.array([[float(v) for v in line.split()] for line in cpp.stdout.splitlines()])
        expected = np.asarray(expected)
        assert actual.shape == expected.shape
        assert np.array_equal(actual[:, :3], expected[:, :3])
        np.testing.assert_allclose(actual[:, 3:], expected[:, 3:], rtol=0, atol=1e-12)
        checks.append('host C++ default kernel matches Python spikes on 20,000 randomized updates')
        csv_path = Path(directory) / 'recordings.csv'
        csv_path.write_text('trial_id,seq,t_us,ax_raw,ay_raw,az_raw,fsr_adc\n'
                            + ''.join(f'check,{i},{i * 5000},0,0,256,1500\n' for i in range(1200)))
        replayed = replay(csv_path)['trials']['check']
        assert replayed['samples'] == 1200
        assert not replayed['baseline_spikes_ms'] and not replayed['proposed_spikes_ms']
        csv_path.write_text('trial_id,seq,t_us,ax_raw,ay_raw,az_raw,fsr_adc\n'
                            'check,0,0,0,0,256,1500\ncheck,2,10000,0,0,256,1500\n')
        try:
            replay(csv_path)
        except ValueError:
            pass
        else:
            raise AssertionError('Sequence gaps must not be silently replayed')
        checks.append('raw-CSV replay accepts a complete synthetic hold and rejects a sequence gap')
    for result in cases.values():
        for key in ['baseline_spike_ticks', 'proposed_spike_ticks']:
            result[key.replace('_ticks', '_ms')] = [tick * 5 for tick in result[key]]
    return dict(status='software_validation_only_not_slip_detection_results',
                physical_recordings_used=0, hardware_tested=False,
                checks_passed=len(checks), checks=checks,
                synthetic_diagnostic_cases=cases,
                cpp_python_parity=dict(updates=20000, spike_mismatches=0,
                                       pressure_tolerance=1e-12, arithmetic='host double'),
                config=config_dict(), real_test_metrics=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate()
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        if args.output.exists():
            raise FileExistsError('Choose a new report file')
        args.output.write_text(payload)
    print(payload)
