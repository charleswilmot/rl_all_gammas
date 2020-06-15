import numpy as np


class ReplayBuffer(object):
    def __init__(self, size):
        self.size = size
        self.current_last = 0

    @staticmethod
    def from_conf(**replay_buffer_conf):
        class_in_conf = eval(replay_buffer_conf.pop("class"))
        return class_in_conf(**replay_buffer_conf)

    def register_episode(self, states, explorative_actions, rewards, dones):
        pass

    def sample(self, batch_size):
        # return in the format that Agent.train accepts
        pass


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(self, size):
        super(SimpleReplayBuffer, self).__init__(size)
        self.dtype = None
        self.sample_index = 0

    def register_episode(self, states, explorative_actions, returns, dones):
        if self.dtype is None: # must create buffer and dtype
            self.dtype = np.dtype([
                ("states", np.float32, states.shape[1:]),
                ("actions", np.float32, explorative_actions.shape[1:]),
                ("returns", np.float32, returns.shape[1:]),
                ("dones", np.bool)
            ])
            self.buffer = np.zeros(self.size, dtype=self.dtype)
        n = states.shape[0]
        indices = self.get_insertion_indices(n)
        if self.current_last < self.size:
            self.current_last = self.current_last + n
        if self.current_last > self.size:
            self.current_last = self.size
        self.buffer["states"][indices] = states
        self.buffer["actions"][indices] = explorative_actions
        self.buffer["returns"][indices] = returns
        self.buffer["dones"][indices] = dones

    def get_insertion_indices(self, n):
        if self.current_last < self.size:
            space_remaining = self.size - self.current_last
            if space_remaining < n:
                # not enough room to insert the full episode
                part1 = np.random.choice(
                    np.arange(self.current_last),
                    n - space_remaining,
                    replace=False
                )
                part2 = np.arange(self.current_last, self.size)
                return np.concatenate((part1, part2))
            else: # enough empty space
                return slice(self.current_last, self.current_last + n)
        else: # buffer already full
            return np.random.choice(np.arange(self.size), n, replace=False)

    def sample(self, batch_size):
        if self.current_last < batch_size or batch_size > self.size:
            return self.buffer[:self.current_last]
        batch_last = self.sample_index + batch_size
        if batch_last < self.current_last:
            ret = self.buffer[self.sample_index:batch_last]
            self.sample_index = batch_last
            return ret
        else: # enough data in buffer but exceed its size
            part1 = self.buffer[self.sample_index:self.current_last]
            part2 = self.buffer[:batch_last - self.current_last]
            self.sample_index = batch_last - self.current_last
            return np.concatenate((part1, part2))
