import numpy as np


class preproces(object):

    def __init__(self, data, focal):

        self.seg_depth = data
        self.Focal = focal
        self.SAMPLE_NUM = 1024
        self.sample_num_l1 = 512
        self.sample_num_l2 = 128

    def run(self):
        """
        run the whole preprocess

        Returns:
            points_cloud (ndarray): the point clouds after all sampling
            max_bb3d_len (float): the max length of the hand
            offset (ndarray): the center point of the hand
        """

        hand_points = self._point_cloud()
        # hand_points_rotate = self._rotate(hand_points)
        hand_points_rotate_sampled = self._sampling(hand_points)
        hand_points_normalized_sampled, max_bb3d_len, offset, location = self._normalize(
            hand_points, hand_points_rotate_sampled)
        (_, points_cloud) = self._two_farthest_sampling(
            hand_points_normalized_sampled)

        return points_cloud, max_bb3d_len, offset, location

    def _point_cloud(self):
        """
        use depth information to generate 3D point clouds

        Returns:
            hand_points (ndarray): the valid point cloud
        """

        [height, width] = self.seg_depth.shape
        pixel_num = width * height

        hand_points = np.empty((pixel_num, 3))
        hand_points[:, 2] = self.seg_depth.reshape(-1)
        valid_idx = hand_points[:, 2] > 0

        w_matrix = np.arange(width, dtype=np.float32)
        for h in range(height):
            hand_points[(h * width):(h + 1) * width, 0] = np.multiply(
                w_matrix - width / 2, self.seg_depth[h, :]) / self.Focal
        h_matrix = np.arange(height, dtype=np.float32)
        for w in range(width):
            idx = [(hi * width + w) for hi in range(height)]
            # whether add "-"
            hand_points[idx, 1] = np.multiply(
                h_matrix - height / 2, self.seg_depth[:, w]) / self.Focal
        # valid_idx = []

        # NOTE: can be repalced by a faster way
        # for num in range(hand_points.shape[0]):
        #     if any(hand_points[num, :]):
        #         valid_idx.append(num)

        hand_points = hand_points[valid_idx, :]

        return hand_points

    # def _rotate(self, hand_points):
    #     """
    #     rotate the hand_points

    #     Args:
    #         hand_points (ndarray): the point clouds of hand

    #     Returns:
    #         hand_points_rotate (ndarray): the rotated hand points of hand
    #     """

    #     pca_mean = np.mean(hand_points, axis=0)
    #     nor = hand_points - pca_mean
    #     (_, _, coeff) = np.linalg.svd(nor, full_matrices=False)
    #     coeff = np.transpose(coeff)
    #     if coeff[1, 0] < 0:
    #         coeff[:, 0] = -coeff[:, 0]
    #     if coeff[2, 2] < 0:
    #         coeff[:, 2] = -coeff[:, 2]
    #     coeff[:, 1] = np.cross(coeff[:, 2], coeff[:, 0])
    #     hand_points_rotate = np.dot(hand_points, coeff)

    #     return hand_points_rotate

    def _sampling(self, hand_points_rotate):
        """
        sampling the point clouds into 1024 points

        Args:
            hand_points_rotate (ndarray): the rotated hand points

        Returns:
            point_cloud_sampling (ndarray): point clouds after sampling
        """

        # hand_points = self._point_cloud()
        point_num = self.SAMPLE_NUM
        point_shape = hand_points_rotate.shape[0]

        if point_shape < point_num:
            rand_idx = np.arange(0, point_num, 1, dtype=np.int32)
            rand_idx[point_shape:] = np.random.randint(0, point_shape,
                                                       size=point_num - point_shape)
        else:
            rand_idx = np.random.randint(0, point_shape, size=point_num)

        point_cloud_sampling = hand_points_rotate[rand_idx, :]

        return point_cloud_sampling  # , rand_idx

    def _normalize(self, hand_points_rotate, hand_points_rotate_sampled):
        """
        normalize the point clouds

        Args:
            hand_points_rotate (ndarray): rotate point clouds
            hand_points_rotate_sampled (ndarray): rotate sampling point clouds

        Returns:
            hand_points_normalized_sampled (ndarray): the point clouds of 1024 points
            max_bb3d_len (float): the max length of the hand
            offset (ndarray): the center point of the hand
        """

        x_min_max = [np.min(hand_points_rotate[:, 0]),
                     np.max(hand_points_rotate[:, 0])]
        y_min_max = [np.min(hand_points_rotate[:, 1]),
                     np.max(hand_points_rotate[:, 1])]
        z_min_max = [np.min(hand_points_rotate[:, 2]),
                     np.max(hand_points_rotate[:, 2])]
        scale = 1.0
        bb3d_x_len = scale*(x_min_max[1]-x_min_max[0])
        bb3d_y_len = scale*(y_min_max[1]-y_min_max[0])
        bb3d_z_len = scale*(z_min_max[1]-z_min_max[0])
        max_bb3d_len = bb3d_x_len

        location = [bb3d_x_len, bb3d_y_len, bb3d_z_len]

        hand_points_normalized_sampled = hand_points_rotate_sampled / max_bb3d_len
        if hand_points_rotate.shape[0] < self.SAMPLE_NUM:
            offset = np.mean(hand_points_rotate, axis=0)/max_bb3d_len
        else:
            offset = np.mean(hand_points_normalized_sampled, axis=0)
        hand_points_normalized_sampled = hand_points_normalized_sampled - offset

        return hand_points_normalized_sampled, max_bb3d_len, offset, location

    def _farthest_point_sampling_fast(self, point_cloud, sample_num):
        """
        sampling point clouds with index order

        Args:
            point_cloud (ndarray): the point clouds using for sampling
            sample_num (int): the selected sampling number, for the best top k numbers

        Returns:
            np.unique(sampled_idx) (ndarray): the unique sampling point clouds index
        """

        pc_num = point_cloud.shape[0]
        if pc_num <= sample_num:
            sampled_idx = np.arange(0, pc_num, 1, dtype=np.int32)
            sampled_idx[pc_num:] = np.random.randint(
                0, pc_num, size=sample_num - pc_num)
        else:
            sampled_idx = np.zeros([sample_num], dtype=np.int32)
            sampled_idx[0] = np.random.randint(1, pc_num)

            cur_sample = point_cloud[sampled_idx[0], :]
            # cur_sample = np.tile(point_cloud[sample_idx[0],:], [sample_num, 1])
            diff = point_cloud - cur_sample
            distance = np.sum(np.multiply(diff, diff), axis=1)

            for i in range(1, sample_num):
                sampled_idx[i] = np.argmax(distance)

                if i < sample_num:
                    valid_idx = distance > 1e-8
                    index_all = np.arange(0, pc_num, 1, dtype=np.int32)
                    valid = index_all[valid_idx]
                    diff = point_cloud[valid, :] - \
                        point_cloud[sampled_idx[i]]
                    new_distance = np.sum(np.multiply(diff, diff), axis=1)
                    # dst = np.vstack((distance[valid], new_distance))
                    distance[valid] = np.min(
                        [distance[valid], new_distance], axis=0)
                    # a = None
        return np.unique(sampled_idx)

    def _two_farthest_sampling(self, hand_points_normalized_sampled):
        """
        do sampling twice

        Args:
            hand_points_normalized_sampled (ndarray): the hand point clouds

        Returns:
             points_cloud_1 (ndarray): point clouds after one sampling, \
              1024 points with the first level 1 number of sampling points
             points_cloud (ndarray): point clouds after two sampling, \
                 1024 points with the two level number of sampling points
        """

        # sample level 1
        sampled_idx_l1 = self._farthest_point_sampling_fast(
            hand_points_normalized_sampled, self.sample_num_l1)

        other_idx = np.array(
            list(set(np.arange(self.SAMPLE_NUM)) - set(sampled_idx_l1)))
        new_idx = np.empty(
            hand_points_normalized_sampled.shape[0], dtype=np.int32)
        new_idx[:sampled_idx_l1.shape[0]] = sampled_idx_l1
        new_idx[sampled_idx_l1.shape[0]:] = other_idx
        points_cloud = hand_points_normalized_sampled[new_idx, :]
        points_cloud_1 = points_cloud.copy()
        # sample level 2
        sampled_idx_l2 = self._farthest_point_sampling_fast(
            points_cloud[:self.sample_num_l1], self.sample_num_l2)
        other_idx = np.array(
            list(set(np.arange(self.sample_num_l1)) - set(sampled_idx_l2)))
        new_idx = np.empty(self.sample_num_l1, dtype=np.int32)
        new_idx[:sampled_idx_l2.shape[0]] = sampled_idx_l2
        new_idx[sampled_idx_l2.shape[0]:] = other_idx
        points_cloud[:self.sample_num_l1, :] = points_cloud[new_idx, :]

        return points_cloud_1, points_cloud
