import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataAnalyzer:
    def __init__(self, input_locations, dims):
        self.x_lim = dims[0]
        self.y_lim = dims[1]
        self.df = pd.DataFrame()
        for file in input_locations:
            df = pd.read_csv(file)
            self.df = pd.concat([self.df, df])

        # self.df = pd.read_csv(input_location)
        self.df = self.df.drop(self.df[(self.df.mocap_x == 0) | (self.df.vision_x == 0)].index)
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
        
        print(axis)
        print(f'Mean: {np.nanmean(data)}')
        print(f'Standard deviation: {np.nanstd(data)}')
        
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




def analyse_axis(df, axis, plot=False):
    data = df[axis].to_numpy()
    timesteps = np.linspace(0, data.size - 1, data.size)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.scatter(timesteps, data, c='blue')
        ax.set_xlabel('timestep')
        ax.set_ylabel(axis)
    
    print(axis)
    print(f'Mean: {(mean := np.nanmean(data))}')
    print(f'Standard deviation: {(std := np.nanstd(data))}')

    return mean, std
    
    # plt.show()

def plot_vector(vec, y_label, x_vec=None, x_label=None, color='blue'):
    data = np.asarray(vec)
    timesteps = np.linspace(0, data.size - 1, data.size) if x_vec is None else x_vec
    x_label = '' if x_label is None else x_label
    # fig = plt.figure()
    # ax = fig.add_subplot()
    plt.scatter(timesteps, data, c=color)
    plt.plot(timesteps, data, color=color, label=y_label)
    # ax.set_xlabel(x_label)
    # ax.set_ylabel(y_label)
    min_val = np.abs(np.min(data))
    max_val = np.abs(np.max(data))
    if min_val > max_val:
        max_val = min_val

    plt.axhline(y=0.0, color='r')
    # ax.set_ylim(-max_val - 0.05 * max_val, max_val + 0.05 * max_val)
    plt.subplots_adjust(left=0.15)
    

    
def analyze_series_3d_error():
    vis = DataAnalyzer(
        [
        'logs/bottle_0_1_1_flying/1.csv',
        'logs/bottle_0_1_1_flying/2.csv',
        'logs/bottle_0_1_1_flying/3.csv',
        'logs/bottle_0_1_1_flying/4.csv',
        'logs/bottle_0_1_1_flying/5.csv',
        'logs/bottle_0_1_1_flying/6.csv',
        'logs/bottle_0_1_1_flying/7.csv',
        'logs/bottle_0_1_1_flying/8.csv',
        ], (480, 640))
        
    # x_vals = [1.7 + i*0.25 for i in range(0,3)]
    x_val = 0.0
    y_vals = [1 + i*0.25 for i in range(0,8)]
    z_val = 1.0
    tolerance = 0.3
    means = []
    stds = []
    axis = ['error_x', 'error_y', 'error_z']
    # axis = ['error_x']
    for val in y_vals:
        # expr_x = abs(abs(vis.df.quad_x - vis.df.mocap_x) - x_val) 
        expr_y = abs(abs(vis.df.quad_y - vis.df.mocap_y) - val) 
        # expr_z = abs(abs(vis.df.quad_z - vis.df.mocap_z) - z_val) 
        
        
        data = expr_y.to_numpy(dtype='float64')
        idx = np.where(data < 0.05)
        data = data[idx]
        print(f'numpy data shape: {data.shape}')
        
        # timesteps = np.linspace(0, data.shape[0] - 1, data.shape[0])
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # plt.scatter(timesteps, data, c='blue')
        # plt.show()


        # temp_df = vis.df[(expr_x < tolerance) & (expr_y < tolerance) & (expr_z < tolerance)]
        temp_df = vis.df[(expr_y < tolerance)]
        print(f'dataframe shape: {temp_df.shape}')
        df_means = []
        df_stds = []
        for a in axis:
            print(val)
            mean, std = analyse_axis(temp_df, a)
            df_means.append(mean)
            df_stds.append(std)
        
        means.append(df_means)
        stds.append(df_stds)

    means_simplified = []
    for entry in means:
        axis_mean = np.mean(entry)
        means_simplified.append(axis_mean)

    means = np.asarray(means)
    x_means = means[:, 0]
    y_means = means[:, 1]
    z_means = means[:, 2]

    fig = plt.figure()
    plot_vector(x_means, 'Mean Error in x [m]', y_vals, 'Translation in x [m]', 'blue')
    plot_vector(y_means, 'Mean Error in y [m]', y_vals, 'Translation in x [m]', 'orange')
    plot_vector(z_means, 'Mean Error in z [m]', y_vals, 'Translation in x [m]', 'magenta')
    plot_vector(means_simplified, 'Combined Mean Error [m]', y_vals, 'Translation in y [m]', 'cyan')
    
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=True)
    plt.xlabel('Translation in x [m]')
    
    fig.set_size_inches(12.0, 8.0, forward=True)
    plt.title('Dynamic Error With Bottle')
    plt.autoscale()
    fig.savefig('dyn_series.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
    # plt.show()
    
if __name__ == '__main__':
    data = DataAnalyzer(['logs/another_perfect_static.csv'], (480, 640))
    quad_x = np.asarray(data.df['quad_x'].astype(float))
    quad_y = np.asarray(data.df['quad_y'].astype(float))
    quad_z = np.asarray(data.df['quad_z'].astype(float))

    quad_roll = np.asarray(data.df['quad_roll'].astype(float))
    quad_pitch = np.asarray(data.df['quad_pitch'].astype(float))
    quad_yaw = np.asarray(data.df['quad_yaw'].astype(float))

    meas_x = np.asarray(data.df['vision_x'].astype(float))

