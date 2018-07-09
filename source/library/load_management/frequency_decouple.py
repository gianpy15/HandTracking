import numpy as np
from threading import Condition


class Interpolator:
    def __init__(self, order: int, samples: int):
        self.order = min(order+1, samples)
        self.samples = samples
        self.values = []
        self.base_time = None
        self.time_features = None
        self.coeff_matrix = None
        self.need_refresh = False
        self.coeff_matrix_lock = Condition()

    def feed(self, time, value):
        lock_taken = False
        if not self.need_refresh or self.coeff_matrix is None:
            self.coeff_matrix_lock.acquire()
            lock_taken = True

        if self.samples > self.current_samples():
            self.values.append((time, value))
            self.base_time = min(self.base_time, time) if self.base_time is not None else time
            self.need_refresh = True
        elif self.base_time < time:
            self.values.append((time, value))
            for i in range(len(self.values)):
                if self.values[i][0] == self.base_time:
                    self.values.remove(self.values[i])
                    break
            self.base_time = min(map(lambda x: x[0], self.values))
            self.need_refresh = True

        if lock_taken:
            if self.need_refresh:
                self.coeff_matrix_lock.notify(n=1)
            self.coeff_matrix_lock.release()

    def refresh_coeff_matrix(self):
        with self.coeff_matrix_lock:
            times = np.array(list(map(lambda x: x[0]-self.base_time, self.values)))
            feats = np.array([times**i for i in range(self.current_order())])
            coeff_shape = (self.current_order(), self.current_samples())
            if np.shape(self.coeff_matrix) != coeff_shape:
                self.coeff_matrix = np.zeros(shape=coeff_shape)
            np.matmul(np.linalg.inv(feats @ feats.T), feats, out=self.coeff_matrix)
            self.need_refresh = False
            self.coeff_matrix_lock.notify_all()

    def current_order(self):
        return min(self.order, self.current_samples())

    def current_samples(self):
        return len(self.values)

    def get(self, time):
        if self.coeff_matrix is None:
            with self.coeff_matrix_lock:
                self.coeff_matrix_lock.wait()
        if self.need_refresh:
            self.refresh_coeff_matrix()
        feats = np.array([(time-self.base_time)**i for i in range(self.current_order())])
        coeffs = np.matmul(feats, self.coeff_matrix)
        return sum(coeffs[i]*self.values[i][1] for i in range(self.current_samples()))

    def __getitem__(self, item):
        return self.get(item)

    def __setitem__(self, key, value):
        self.feed(time=key, value=value)


if __name__ == '__main__':
    import matplotlib.pyplot as mplt
    f = Interpolator(order=3, samples=6)

    datax = [0.5, 1.1, 2.0, 1.0, 4.1, 2.9, 3.5]
    datay = [1.0, 2.2, 4.0, 2.1, 5.3, 3.1, 2.7]

    for (x, y) in zip(datax, datay):
        f[x] = y

    mplt.plot(datax, datay, 'ro')
    x = np.arange(max(datax)*1.25, step=0.1)
    y = np.array([f[xi] for xi in x])
    mplt.plot(x, y, 'b-')
    mplt.show()
