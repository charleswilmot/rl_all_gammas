import numpy as np
import matplotlib.pyplot as plt


plt.ion()


class ReturnViewer(object):
    def __init__(self, lookback=100):
        self._fig = plt.figure()
        self._fig_shawn = False
        self._ax = self._fig.add_subplot(111)
        self._return_line = None
        self._estimated_return_line = None
        self._lookback = lookback


    def __call__(self, returns, estimated_returns):
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
            mini = min(np.min(returns), np.min(estimated_returns))
            maxi = max(np.max(returns), np.max(estimated_returns))
            self._ax.set_ylim([mini - 0.1, maxi + 0.1])
        if self._fig_shawn:
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
        else:
            self._fig.show()
            self._fig_shawn = True
            self._fig.canvas.flush_events()


if __name__ == '__main__':
    import time

    plt.ion()
    rv = ReturnViewer()

    if rv:
        print("yeeeeeees")

    for i in range(10):
        rv(
            np.random.uniform(size=50),
            np.random.uniform(size=50)
        )
        time.sleep(1)
