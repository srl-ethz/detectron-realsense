import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataAnalyzer:
    def __init__(self, input_location, dims):
        self.x_lim = dims[0]
        self.y_lim = dims[1]
        self.df = pd.read_csv(input_location)
        self.avg_fps = 0
        self.avg_position = (0, 0)

    def export_to_csv(self, output_location):
        self.df.to_csv(output_location)

    def add_fps_to_df(self):
        timestamps = self.df['t'].to_numpy()
        fps = [0]

        for i in range(1, len(timestamps)):
            if timestamps[i] - timestamps[i-1] != 0:
                fps.append(1/(timestamps[i] - timestamps[i-1]))
            else:
                fps.append(fps[i-1])

        self.df.insert(len(self.df.columns), 'fps', fps)

    def visualize_fps_raw(self):
        if 'fps' not in self.df.columns:
            self.add_fps_to_df()

        fps = self.df['fps'].to_numpy()
        timesteps = np.linspace(0, fps.size - 1, fps.size)
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.scatter(timesteps, fps, c='blue')
        ax.set_xlabel('frames')
        ax.set_ylabel('fps [1/s]')
        print(np.average(fps))
        plt.show()

    def visualize_axis_raw(self, axis):
        data = self.df[axis].to_numpy()
        timesteps = np.linspace(0, data.size - 1, data.size)
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.scatter(timesteps, data, c='blue')
        ax.set_xlabel('timestep')
        ax.set_ylabel(axis)
        plt.show()

    def visualize_3d_pixels(self):
        x_vec = self.df['x'].to_numpy()
        y_vec = self.df['y'].to_numpy()
        z_vec = self.df['z'].to_numpy()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for i in range(0, len(x_vec)):
            # swap y and z axis in visualisation
            ax.scatter(x_vec[i], z_vec[i], y_vec[i], c='blue')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        ax.set_zlabel('y [m]')

        z_lim = np.max(z_vec)
        ax.set_xlim((0, self.x_lim))
        ax.set_ylim((0, z_lim))
        ax.set_zlim((0, self.y_lim))

        plt.show()

    def visualize_3d_meters(self):
        x_vec = self.df['x'].to_numpy()
        y_vec = self.df['y'].to_numpy()
        z_vec = self.df['z'].to_numpy()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for i in range(0, len(x_vec)):
            # swap y and z axis in visualisation
            ax.scatter(x_vec[i], z_vec[i], y_vec[i], c='blue')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        ax.set_zlabel('y [m]')

        x_lim = (np.min(x_vec), np.max(x_vec))
        y_lim = (np.min(z_vec), np.max(z_vec))
        z_lim = (np.min(y_vec), np.max(y_vec))

        ax.set_xlim((x_lim[0], x_lim[1]))
        ax.set_zlim((z_lim[0], z_lim[1]))
        ax.set_ylim((y_lim[0], y_lim[1]))

        plt.show()

    def visualize_2D_pixels(self):
        x_vec = self.df['x'].to_numpy()
        y_vec = self.df['y'].to_numpy()

        fig = plt.figure()
        ax = fig.add_subplot()

        for i in range(0, len(x_vec)):
            ax.scatter(x_vec[i], y_vec[i], c='blue')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        ax.set_xlim((0, self.x_lim))
        ax.set_ylim((0, self.y_lim))

        plt.show()


if __name__ == '__main__':
    vis = DataAnalyzer(
        'logs/test_8.csv', (480, 640))
    # vis.add_fps_to_df()
    vis.visualize_fps_raw()
    # vis.visualize_3d_meters()
    # vis.visualize_axis_raw('z')
