import numpy as np
import math

class NaiveDecay(object):

    def __init__(self, initial_eps, episode_shrinkage, limiting_epsiode, change_after_n_episodes):
        self.initial_eps = initial_eps
        self.episode_shrinkage = episode_shrinkage
        self.limiting_epsiode = limiting_epsiode
        self.change_after_n_episodes = change_after_n_episodes

    def apply(self, episode, _):
        if episode >= self.limiting_epsiode:
            return 0
        eps = self.initial_eps-(self.episode_shrinkage * (episode//self.change_after_n_episodes))
        eps = min(max(eps, 0), 1)
        return eps

class Constant(object):

    def __init__(self, value):
        self.value = value

    def apply(self, *args):
        return self.value


class GaussianDecay(object):

    def __init__(self, final, scaling, offset, max_steps):
        self.final = final
        self.scaling = scaling
        self.offset = offset
        self.weight = math.sqrt(-math.log(final)) / max_steps

    def apply(self, episode, _):
        return math.exp(-(self.weight * episode) ** 2) * self.scaling + self.offset


class ExpSawtoothEpsDecay(object):

    def __init__(self, initial_eps, episode_shrinkage, step_increase, limiting_epsiode, change_after_n_episodes):
        self.initial_eps = initial_eps
        self.episode_shrinkage = episode_shrinkage
        self.step_increase = step_increase
        self.limiting_epsiode = limiting_epsiode
        self.change_after_n_episodes = change_after_n_episodes

    def apply(self, episode, _):
        if episode >= self.limiting_epsiode:
            return 0
        eps = np.exp(-(0.0025 * episode) ** 2)
        # eps = eps + (self.step_increase * step / (episode+1))
        eps = min(max(eps, 0), 1)
        return eps


class RunningAverage(object):
    def __init__(self, weights, band_width=10, init_val=0, offset=0):
        super(RunningAverage, self).__init__()
        self.band_width = band_width
        self.init_val = init_val
        self._mem = [init_val]
        self.offset = offset
        self.weights = weights
        self._avg = init_val

    def reset(self):
        self._mem = [self.init_val]

    def apply(self, el):
        if len(self._mem) == self.band_width:
            self._mem.pop(0)
        self._mem.append(el)
        if len(self._mem) == self.band_width:
            self._avg = (np.array(self._mem) * self.weights).sum() + self.offset
            return self._avg
        return None

    @property
    def avg(self):
        return self._avg


class ExponentialAverage(object):
    def __init__(self, base, weight, init_val):
        super(ExponentialAverage, self).__init__()
        self._weight = weight
        self._init_val = init_val
        self._state = init_val
        self._base = base

    def reset(self):
        self._state = self._init_val

    def apply(self, _, el):
        el = min(1, (el / self._base))
        self._state = self._weight * self._state + (1 - self._weight) * el
        return self._state

class FollowLeadAvg(object):
    def __init__(self, base_val, band_width, init_val):
        super(FollowLeadAvg, self).__init__()
        self.base = base_val
        self.band_width = band_width
        self.init_val = init_val
        self._mem = [init_val]

    def reset(self):
        self._mem = [self.init_val]

    def apply(self, _, el):
        if len(self._mem) == self.band_width:
            self._mem.pop(0)
        self._mem.append(min(self.init_val, (el / self.base)))
        return self.avg

    @property
    def avg(self):
        return sum(self._mem) / len(self._mem)


class FollowLeadMin(object):
    def __init__(self, base_val, init_val):
        super(FollowLeadMin, self).__init__()
        self._base = base_val
        self._init_val = init_val
        self._current = np.inf

    def reset(self):
        self._current = np.inf

    def apply(self, _, el):
        if el < self._current:
            self._current = el
        return min(1, (self._current / self._base))


class ActionPathTreeNodes(object):

    def __init__(self):
        self.memory = {}

    def push_path(self, path):
        if path != "":
            key = path
        else:
            key = "first"
        if key in self.memory:
            self.memory[key] += 1
        else:
            self.memory[key] = 1

    def get_n_visits(self, path):
        if path != "":
            key = path
        else:
            key = "first"
        if key in self.memory:
            return self.memory[key]
        else:
            return 0

    def set_n_visits(self, path, visits):
        if path != "":
            key = path
        else:
            key = "first"
        self.memory[key] = visits

    def clear_memory(self):
        self.memory = {}