from collections import namedtuple
import random
import numpy as np
import torch
from multiprocessing import Lock

Transition_t = namedtuple('Transition', ('state', 'actions', 'reward', 'state_', 'time', 'behav_probs', 'terminal'))

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'terminal'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

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

    def __reduce__(self):
        return (self.__class__, (self._cap, ))

    def __len__(self):
        return len(self._memory)

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
        finally:
            self._mtx.release()

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
        self._memory = []
        self._sample_counts = []
        self._losses = []
        self._position = 0

    def __len__(self):
        return len(self._memory)

    def __del__(self):
        del self._mtx


class TransitionData(object):

    def __init__(self, capacity, storage_object):
        self.memory = []
        self.position = 0
        self.cap = capacity
        self.storage_object = storage_object

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        """Saves a transition."""
        if self.position >= self.cap:
            self.pop(0)
        self.memory.append(None)
        self.memory[self.position] = self.storage_object(*args)
        self.position += 1

    def pop(self, position):
        self.position -= 1
        return self.memory.pop(position)

    def sample(self, batch_size, distribution=None):
        return np.random.choice(self.memory, size=batch_size, p=distribution)

    def clear(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)


# -*- coding: utf-8 -*-
import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'policy'))


class EpisodeTrajectoriesMem():
  def __init__(self, capacity, max_episode_length):
    # Max number of transitions possible will be the memory capacity, could be much less
    self.num_episodes = capacity
    self.memory = deque(maxlen=self.num_episodes)
    self.trajectory = []

  def append(self, state, action, reward, policy):
    self.trajectory.append(Transition(state, action, reward, policy))  # Save s_i, a_i, r_i+1, µ(·|s_i)
    # Terminal states are saved with actions as None, so switch to next episode
    if action is None:
      self.memory.append(self.trajectory)
      self.trajectory = []
  # Samples random trajectory
  def sample(self, maxlen=0):
    mem = self.memory[random.randrange(len(self.memory))]
    T = len(mem)
    # Take a random subset of trajectory if maxlen specified, otherwise return full trajectory
    if maxlen > 0 and T > maxlen + 1:
      t = random.randrange(T - maxlen - 1)  # Include next state after final "maxlen" state
      return mem[t:t + maxlen + 1]
    else:
      return mem

  def sample_batch(self, batch_size, maxlen=0):
    batch = [self.sample(maxlen=maxlen) for _ in range(batch_size)]
    return list(map(list, zip(*batch)))  # Transpose so that timesteps are packed together

  def length(self):
    # Return number of epsiodes saved in memory
    return len(self.memory)

  def __len__(self):
    return sum(len(episode) for episode in self.memory)


if __name__ == "__main__":
    import pickle

    print(pickle.dumps(TransitionData_ts(1,1)))