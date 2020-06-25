import numpy as np


class ReplayBufferBase(object):
    def __init__(self, size):
        self.size = size
        self.current_last = 0
        self.is_prioritized = False

    @staticmethod
    def from_conf(**replay_buffer_conf):
        class_in_conf = eval(replay_buffer_conf.pop("class"))
        return class_in_conf(**replay_buffer_conf)


def type_mapper(value):
    if type(value) is np.ndarray:
        return value.dtype
    elif type(value) is list:
        if type(value[0]) is int:
            return np.int32
        elif type(value[0]) is float:
            return np.float32
        elif type(value[0]) is bool:
            return np.bool
        else:
            raise ValueError("Unrecognized type: {}".format(type(value)))
    else:
        raise ValueError("Must be a list or an array ({})".format(type(value)))


def shape_mapper(value):
    if type(value) is np.ndarray:
        return value.shape[1:]  # skip batch dimension
    elif type(value) is list:
        return ()
    else:
        raise ValueError("Must be a list or an array ({})".format(type(value)))


class ReplayBuffer(ReplayBufferBase):
    def __init__(self, size):
        super(ReplayBuffer, self).__init__(size)
        self.dtype = None
        self.sample_index = 0

    def _contruct_dtype(self, **data):
        self.dtype = np.dtype([
            (key, type_mapper(value), shape_mapper(value))
            for key, value in data.items()
        ])

    def _construct_buffer(self):
        self.buffer = np.zeros(self.size, dtype=self.dtype)

    def register_episode(self, **data):
        if self.dtype is None: # must create buffer and dtype
            self._contruct_dtype(**data)
            self._construct_buffer()
        n = len(next(iter(data.values())))
        indices = self.get_insertion_indices(n)
        if self.current_last < self.size:
            self.current_last = self.current_last + n
        if self.current_last > self.size:
            self.current_last = self.size
        for key, value in data.items():
            self.buffer[key][indices] = value

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


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, beta):
        super(PrioritizedReplayBuffer, self).__init__(size)
        self.alpha = alpha  # 0.6
        self.beta = beta    # 0.4
        self.is_prioritized = True

    def _contruct_dtype(self, **data):
        self.dtype = np.dtype([
            (key, type_mapper(value), shape_mapper(value))
            for key, value in data.items()
        ] + [
            ("indices", np.int32),
            ("importance", np.float32),
            ("probabilities", np.float32)
        ])

    def _construct_buffer(self):
        self.buffer = np.zeros(self.size, dtype=self.dtype)
        self.buffer["indices"] = np.arange(self.size)
        self.buffer["importance"] = 1

    def register_episode(self, priorities, **data):
        # enforce a field 'priorities'
        super(PrioritizedReplayBuffer, self).register_episode(
            priorities=priorities,
            **data
        )

    def sample(self, batch_size):
        if self.current_last < batch_size or batch_size > self.size:
            return self.buffer[:self.current_last]
        # prob = p_k^alpha / sum_k p_k^alpha
        self.buffer["probabilities"][:self.current_last] = \
                    self.buffer["priorities"][:self.current_last] ** self.alpha
        self.buffer["probabilities"][:self.current_last] /= \
                    np.sum(self.buffer["probabilities"][:self.current_last])
        indices = np.random.choice(
            self.current_last,
            batch_size,
            replace=False,
            p=self.buffer["probabilities"][:self.current_last]
        )
        # importance = (batch_size * prob_k) ** -beta
        self.buffer["importance"][indices] = \
            (batch_size * self.buffer["probabilities"][indices]) ** -self.beta
        return self.buffer[indices]

    def update_priorities(self, indices, priorities):
        self.buffer["priorities"][indices] = priorities


if __name__ == "__main__":
    def test_simple_replay_buffer():
        replay = ReplayBuffer(16)

        all_returns = np.arange(100, dtype=np.float32).reshape((20, 5, 1))

        for returns in all_returns:
            replay.register_episode(returns=returns)
            print(replay.buffer)
            print("")
            print(replay.sample(5))
            print("")
            print(replay.sample(5))
            print("\n\n\n")

    test_simple_replay_buffer()
