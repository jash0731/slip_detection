import numpy as np


def chronological_order(records):
    return sorted(range(len(records)), key=lambda i: (records[i]['group'], records[i]['end_sample']))


def positive_runs(y, records):
    runs, run, last = [], [], None
    for i in chronological_order(records):
        r = records[i]
        contiguous = (last is not None and r['group'] == last['group']
                      and r['end_sample'] - last['end_sample'] == 16)
        if not contiguous or y[i] != 1:
            if run:
                runs.append(run)
                run = []
        if y[i] == 1:
            run.append(i)
        last = r
    if run:
        runs.append(run)
    return runs


def smoothed_scores(scores, records, alpha):
    result = np.zeros(len(scores))
    last, state = None, 0.
    for i in chronological_order(records):
        r = records[i]
        reset = (last is None or r['group'] != last['group']
                 or r['end_sample'] - last['end_sample'] != 16)
        state = scores[i] if reset else alpha * scores[i] + (1 - alpha) * state
        result[i] = state
        last = r
    return result


def bypass_decisions(scores, records, alpha, gap, margin, thresholds):
    thresholds = np.asarray(thresholds)
    smoothed = smoothed_scores(scores, records, alpha)
    result = np.zeros((len(scores), len(thresholds)), dtype=bool)
    state = np.zeros(len(thresholds), dtype=bool)
    last = None
    for i in chronological_order(records):
        r = records[i]
        reset = (last is None or r['group'] != last['group']
                 or r['end_sample'] - last['end_sample'] != 16)
        if reset:
            state[:] = False
        state = np.where(state, smoothed[i] >= thresholds - gap / 2,
                         smoothed[i] >= thresholds + gap / 2)
        bypass = False if margin is None else scores[i] >= thresholds + margin
        result[i] = state | bypass
        last = r
    return result


def add_only_decisions(scores, records, alpha, guard, thresholds, current_threshold):
    thresholds = np.asarray(thresholds)
    smoothed = smoothed_scores(scores, records, alpha)
    return ((scores >= current_threshold)[:, None]
            | ((smoothed[:, None] >= thresholds[None, :])
               & (scores[:, None] >= current_threshold - guard)))


class TemporalAlarm:
    def __init__(self, choice, current_threshold):
        self.choice = dict(choice)
        self.current_threshold = current_threshold
        self.last = None
        self.smoothed = 0.
        self.alarm = False

    def step(self, score, group, end_sample):
        c = self.choice
        reset = (self.last is None or group != self.last[0]
                 or end_sample - self.last[1] != 16)
        if reset:
            self.smoothed = score
            self.alarm = False
        else:
            self.smoothed = c['alpha'] * score + (1 - c['alpha']) * self.smoothed
        self.last = (group, end_sample)
        if c['kind'] == 'add_only':
            extra = (c['threshold'] is not None and self.smoothed >= c['threshold']
                     and score >= self.current_threshold - c['guard'])
            return bool(score >= self.current_threshold or extra)
        threshold = c['threshold'] - c['gap'] / 2 if self.alarm else c['threshold'] + c['gap'] / 2
        self.alarm = self.smoothed >= threshold
        bypass = c['margin'] is not None and score >= c['threshold'] + c['margin']
        return bool(self.alarm or bypass)
