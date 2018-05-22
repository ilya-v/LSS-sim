from layout import *


class LeapFrogIntegrator:
    def __init__(self):
        self.time_step = 0.0

    def begin_step(self, X):
        X[:, QVEL] += X[:, QACC] * self.time_step / 2
        X[:, QCOORD] += X[:, QVEL] * self.time_step

    def end_step(self, X):
        X[:, QVEL] += X[:, QACC] * self.time_step / 2


class SimplifiedLeapFrogIntegrator:
    def __init__(self):
        self.time_step = 0.0

    def begin_step(self, X):
        pass

    def end_step(self, X):
        X[:, QVEL] += X[:, QACC] * self.time_step
        X[:, QCOORD] += X[:, QVEL] * self.time_step
