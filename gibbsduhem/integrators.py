import numpy as np


class Integrator:
    """Base integrator class. The implemented integrators should
    be able to solve the differential equation

    dy
    -- = f(t, y)
    dt

    efficiently.

    :param dt: step length
    :type dt: float
    :param f: f(t, y) is a function of t and y
    :type f: func
    """
    def __init__(self, dt, f):
        self.dt = dt
        self.f = f

    def find_next(self, t, y):
        """Find next y-value, given the previous.
        """
        pass


class ForwardEuler(Integrator):
    """The forward Euler integration scheme solves the differential
    equation merely by using the previous value of f. This is among the
    cheapest integration methods available, but performs accordingly.
    """
    def find_next(self, t, y):
        return t+self.dt, y+self.f(t, y)*self.dt


class PredictorCorrector(Integrator):
    """The Predictor-Corrector integration scheme solves the differential
    equation by first predicting the next y-value, then updating f(x, y_new)
    and finally correcting the estimated y-value.
    """
    def find_next(self, t, y):
        f_0 = self.f(t, y)
        y_pred = y + f_0
        f_1 = self.f(t, y_pred)
        return t+self.dt, y+(f_0+f_1)*self.dt/2


class RungeKutta2(Integrator):
    """RungeKutta2 is a second-order integration method.
    """
    def find_next(self, t, y):
        return t+self.dt, y+self.f(t, y+self.dt/2)


class RungeKutta4(Integrator):
    """RungeKutta4 is a fourth-order integration method.
    """
