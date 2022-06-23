import filterpy
import numpy as np

class Filterer:
    def __init__(self, initial_estimate, initial_cov_matrix) -> None:
        self.filter = filterpy.KalmanFilter(dim_x=4, dim_z=3)
        self.filter.x = np.asarray(initial_estimate)
        self.filter.H = np.asarray([1,0,0,0],
                                    [0,1,0,0],
                                    [0,0,1,0])
        self.filter.R = initial_cov_matrix
        self.filter.Q = filterpy.common.Q_discrete_white_noise(dim=4, dt=0.1, var=0.05)
        self.filter.P *= 0.05

    def update_and_get_state_prediction(self, measurement, state_trans_matrix):
        self.filter.F = state_trans_matrix
        self.filter.predict()
        pred_state = self.filter.x
        self.filter.update(measurement)

        return pred_state


