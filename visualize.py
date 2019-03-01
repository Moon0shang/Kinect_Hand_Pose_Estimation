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
    depth = np.load('./sample.npy')
    depth = {'depth': depth}
    pp = preproces(depth)
    hand_points = pp._point_cloud()

    idx = hand_points[:, 2] > 500
    hand_points = hand_points[idx, :]
    visualize(hand_points, "seg_hand")

    hand_points_rotate = pp._rotate(hand_points)
    visualize(hand_points_rotate, "rotate")

    hand_points_sampling = pp._sampling(hand_points_rotate)
    visualize(hand_points_sampling, "sampling")

    hand_points_normalized = pp._normalize(
        hand_points_rotate, hand_points_sampling)
    visualize(hand_points_sampling, "normalized")

    points_cloud_1, points_cloud = pp.get_point_cloud(hand_points_normalized)
    visualize(points_cloud_1, "PC1")
    visualize(points_cloud, "PC2")

    plt.ioff()
    plt.show()
