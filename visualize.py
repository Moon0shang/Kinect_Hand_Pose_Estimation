import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize(hand_points, label):
    x = hand_points[:, 0]
    y = hand_points[:, 1]
    z = hand_points[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        x, y, z,
        marker='.'  # show as a point
    )
    ax.axis('scaled')
    ax.set_title('%s' % label)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.pause(0.5)


if __name__ == "__main__":

    plt.ion()
    data_matlab = sio.loadmat('./matlab/hand_point-1.mat')
    data_matlab = data_matlab['hand_points']
    visualize(data_matlab, 'matlab')
    data_python = sio.loadmat('./python/hand_points-1.mat')
    data_python = data_python['hand_points']
    visualize(data_python, 'python')
    plt.ioff()
    plt.show()
