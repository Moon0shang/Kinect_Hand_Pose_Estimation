import numpy as np
from preprocess import preproces
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

    Focal = 364.5
    depth = np.load('./results/pc/points_cloud100.npy')
    pre = preproces(depth, Focal)
    point_clouds, max_bb3d_len, offset, location = pre.run()
    visualize(point_clouds, 'point cloud')
    visualize(pc2[:512, :], "level 1")
    visualize(pc2[:128, :], "level 2")

    plt.ioff()
    plt.show()
