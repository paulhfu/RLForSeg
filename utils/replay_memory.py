from collections import namedtuple
import random
import numpy as np
import torch
from multiprocessing import Lock, Event

Transition_ts = namedtuple('Transition', ('state', 'action', 'reward'))
class TransitionData_ts(object):

    def __init__(self, capacity):
        self._memory = []
        self._position = 0
        self._cap = capacity
        self._sample_counts = []
        self._losses = []
        self._max_loss = 1e-10
        self._mtx = Lock()
        self.is_full_event = Event()
        self.reset_event = Event()
        self.reset_event.set()
        self._push_count = 0

    def __reduce__(self):
        return (self.__class__, (self._cap, ))

    def __len__(self):
        return len(self._memory)

    @property
    def push_count(self):
        return self._push_count

    def reset_push_count(self):
        self._mtx.acquire()
        self._push_count = 0
        self._mtx.release()

    def push(self, *args):
        """Saves a transition."""
        self._mtx.acquire()
        try:
            if self._position >= self._cap:
                drop_out = self._get_joint_dis().argmin()
                self._pop(drop_out)
            self._memory.append(None)
            self._sample_counts.append(1)
            self._losses.append(self._max_loss)
            self._memory[self._position] = Transition_ts(*args)
            self._position += 1
            self._push_count += 1
        finally:
            self._mtx.release()
        if self._position >= self._cap and not self.is_full_event.is_set():
            self.is_full_event.set()

    def is_full(self):
        if self._position >= self._cap:
            return True
        return False

    def _get_joint_dis(self):
        sample_counts = torch.tensor(self._sample_counts, dtype=torch.float)
        losses = torch.tensor(self._losses, dtype=torch.float)
        ret = 1 / sample_counts
        ret = ret / ret.max() + 4 * losses / losses.max()
        return ret

    def _pop(self, position):
        self._position -= 1
        self._sample_counts.pop(position)
        self._losses.pop(position)
        return self._memory.pop(position)

    def sample(self):
        self._mtx.acquire()
        try:
            distribution = torch.softmax(self._get_joint_dis(), 0)
            sample_idx = torch.multinomial(distribution, 1).item()
            self._sample_counts[sample_idx] += 1
        finally:
            self._mtx.release()
        return self._memory[sample_idx], sample_idx

    def report_sample_loss(self, loss_val, sample_index):
        self._mtx.acquire()
        try:
            if torch.isinf(loss_val) or torch.isnan(loss_val):
                self._losses[sample_index] = self._max_loss
            else:
                if loss_val > self._max_loss:
                    self._max_loss = loss_val
                self._losses[sample_index] = loss_val
        finally:
            self._mtx.release()

    def clear(self):
        self._mtx.acquire()
        try:
            self._memory = []
            self._sample_counts = []
            self._losses = []
            self._position = 0
        finally:
            self._mtx.release()

    def __len__(self):
        return len(self._memory)

    def __del__(self):
        del self._mtx


class TransitionData_episodes(object):

    def __init__(self, capacity):
        self._memory = []
        self._position = 0
        self._cap = capacity
        self._sample_counts = []
        self._losses = []
        self._max_loss = 1e-10
        self._mtx = Lock()
        self.is_full_event = Event()
        self.reset_event = Event()
        self.reset_event.set()
        self._push_count = 0
        self.episode = []

    def __reduce__(self):
        return (self.__class__, (self._cap, ))

    def __len__(self):
        return len(self._memory)

    @property
    def push_count(self):
        return self._push_count

    def reset_push_count(self):
        self._mtx.acquire()
        self._push_count = 0
        self._mtx.release()

    def push(self, episode):
        """Saves a transition."""
        self._mtx.acquire()
        try:
            if self._position >= self._cap:
                drop_out = self._get_joint_dis().argmin()
                self._pop(drop_out)
            self._memory.append(None)
            self._sample_counts.append(1)
            self._losses.append(self._max_loss)
            self._memory[self._position] = episode
            self._position += 1
            self._push_count += 1
        finally:
            self._mtx.release()
        if self._position >= self._cap and not self.is_full_event.is_set():
            self.is_full_event.set()

    def is_full(self):
        if self._position >= self._cap:
            return True
        return False

    def _get_joint_dis(self):
        sample_counts = torch.tensor(self._sample_counts, dtype=torch.float)
        losses = torch.tensor(self._losses, dtype=torch.float)
        ret = 1 / sample_counts
        ret = ret / ret.max() + 4 * losses / losses.max()
        return ret

    def _pop(self, position):
        self._position -= 1
        self._sample_counts.pop(position)
        self._losses.pop(position)
        return self._memory.pop(position)

    def sample(self):
        self._mtx.acquire()
        try:
            distribution = torch.softmax(self._get_joint_dis(), 0)
            sample_idx = torch.multinomial(distribution, 1).item()
            self._sample_counts[sample_idx] += 1
        finally:
            self._mtx.release()
        return self._memory[sample_idx], sample_idx

    def report_sample_loss(self, loss_val, sample_index):
        self._mtx.acquire()
        try:
            if torch.isinf(loss_val) or torch.isnan(loss_val):
                self._losses[sample_index] = self._max_loss
            else:
                if loss_val > self._max_loss:
                    self._max_loss = loss_val
                self._losses[sample_index] = loss_val
        finally:
            self._mtx.release()

    def clear(self):
        self._mtx.acquire()
        try:
            self._memory = []
            self._sample_counts = []
            self._losses = []
            self._position = 0
        finally:
            self._mtx.release()

    def __len__(self):
        return len(self._memory)

    def __del__(self):
        del self._mtx

if __name__ == "__main__":
    import pickle

    print(pickle.dumps(TransitionData_ts(1,1)))