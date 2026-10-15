import argparse
import csv
import json
from pathlib import Path

from .model import Config, Kernel, Preprocessor, merge_events


def replay(path, config=Config()):
    trials, states = {}, {}
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'trial_id', 'seq', 't_us', 'ax_raw', 'ay_raw', 'az_raw', 'fsr_adc'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Recording does not contain the protocol sensor columns')
        for row in reader:
            trial = row['trial_id']
            if trial not in states:
                states[trial] = (Preprocessor(config), Kernel(config, False), Kernel(config, True), None)
                trials[trial] = dict(samples=0, first_t_us=int(row['t_us']),
                                     baseline_spikes_ms=[], proposed_spikes_ms=[])
            pre, baseline, proposed, previous = states[trial]
            seq, timestamp = int(row['seq']), int(row['t_us'])
            if previous is not None:
                if seq != previous[0] + 1 or timestamp <= previous[1]:
                    raise ValueError(f'Sequence gap or nonmonotonic time in {trial}')
                if abs(timestamp - previous[1] - 5000) > 1000:
                    raise ValueError(f'Sample spacing outside the declared 4-6 ms replay range in {trial}')
            raw = [float(row[key]) for key in ['ax_raw', 'ay_raw', 'az_raw']]
            events = pre.step([value * config.acceleration_lsb_g for value in raw], float(row['fsr_adc']))
            b, p = baseline.step(*events), proposed.step(*events)
            for name, result in [('baseline', b), ('proposed', p)]:
                if result['spike']:
                    trials[trial][name + '_spikes_ms'].append(timestamp / 1000.)
            trials[trial]['samples'] += 1
            states[trial] = (pre, baseline, proposed, (seq, timestamp))
    if not trials:
        raise ValueError('No recordings found')
    for result in trials.values():
        for name in ['baseline', 'proposed']:
            result[name + '_warning_events_ms'] = merge_events(result[name + '_spikes_ms'])
    return dict(status='replay_only_requires_independent_video_labels_for_scoring', trials=trials)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('recordings', type=Path)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    cfg = Config(**json.loads(args.config.read_text())) if args.config else Config()
    print(json.dumps(replay(args.recordings, cfg), indent=2, allow_nan=False))
