import numpy as np
import matplotlib.pyplot as plt
from time import time


plt.ion()


class ReturnViewer(object):
    def __init__(self, lookback=500, min_refresh_interval=0.5):
        self._fig = plt.figure()
        self._fig_shawn = False
        self._ax = self._fig.add_subplot(111)
        self._return_line = None
        self._estimated_return_line = None
        self._lookback = lookback
        self._min_refresh_interval = min_refresh_interval
        self._last_refresh_timestamp = time()
        self._min_y_lim = None
        self._max_y_lim = None
        self._lim_decay = 0.95

    def __call__(self, returns, estimated_returns):
        timestamp = time()
        interval = timestamp - self._last_refresh_timestamp
        if interval < self._min_refresh_interval:
            return
        else:
            self._last_refresh_timestamp = timestamp
        size = returns.shape[0]
        if size < self._lookback:
            returns = np.pad(
                returns,
                (self._lookback - size, 0),
                constant_values=returns[0]
            )
            estimated_returns = np.pad(
                estimated_returns,
                (self._lookback - size, 0),
                constant_values=estimated_returns[0]
            )
        returns = returns[-self._lookback:]
        estimated_returns = estimated_returns[-self._lookback:]
        if self._return_line is None:
            X = np.arange(-len(returns), 0)
            self._return_line, = self._ax.plot(
                X, returns, linewidth=2,
                label="target"
            )
            self._estimated_return_line, = self._ax.plot(
                X, estimated_returns, linewidth=2,
                label="estimate"
            )
            self._ax.set_title("Return quality")
            self._ax.set_ylabel("Return")
            self._ax.set_xlabel("#Iteration")
            self._ax.legend()
        else:
            self._return_line.set_ydata(returns)
            self._estimated_return_line.set_ydata(estimated_returns)
            mini = min(np.min(returns), np.min(estimated_returns)) - 0.1
            maxi = max(np.max(returns), np.max(estimated_returns)) + 0.1
            if self._min_y_lim is None:
                self._min_y_lim = mini
            if self._max_y_lim is None:
                self._max_y_lim = maxi
            self._min_y_lim = min(self._min_y_lim * self._lim_decay, mini)
            self._max_y_lim = max(self._max_y_lim * self._lim_decay, maxi)
            self._ax.set_ylim([self._min_y_lim, self._max_y_lim])
        if self._fig_shawn:
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
        else:
            self._fig.show()
            self._fig_shawn = True
            self._fig.canvas.flush_events()


if __name__ == '__main__':
    from time import sleep

    # plt.ion()
    rv = ReturnViewer()

    if rv:
        print("yeeeeeees")

    for i in range(1000):
        rv(
            np.random.uniform(size=500),
            np.random.uniform(size=500)
        )
        sleep(0.1)
        if i % 100 == 99:
            print("pause start")
            sleep(10)
            print("pause stop")
