import numpy as np
import scipy.io as sio


class preproces(object):

    def __init__(self, data):
        # test variable
        self.fFocal_msra = 241.42
        self.data = data
        #self.point_num = point_num
        # real variable
        self.seg_depth = data['depth']
        self.Focal = 365.45
        self.SAMPLE_NUM = 1024
        self.sample_num_l1 = 512
        self.sample_num_l2 = 128

    def run(self):
        hand_points = self._point_cloud()
        hand_points_rotate = self._rotate(hand_points)
        hand_points_rotate_sampled = self._sampling(hand_points_rotate)
        hand_points_normalized_sampled = self._normalize(
            hand_points_rotate, hand_points_rotate_sampled)
        (_, points_cloud) = self._two_farthest_sampling(
            hand_points_normalized_sampled)

        return points_cloud

    def _point_cloud(self):

        [height, width] = self.seg_depth.shape
        pixel_num = width * height

        hand_points = np.empty((pixel_num, 3))
        hand_points[:, 2] = self.seg_depth.reshape(-1)

        w_matrix = np.arange(width, dtype=np.float32)
        for h in range(height):
            hand_points[(h * width):(h + 1) * width, 0] = np.multiply(
                w_matrix - width / 2, self.seg_depth[h, :]) / self.Focal
        h_matrix = np.arange(height, dtype=np.float32)
        for w in range(width):
            idx = [(hi * width + w) for hi in range(height)]
            # whether add "-"
            hand_points[idx, 1] = -np.multiply(
                h_matrix - height / 2, self.seg_depth[:, w]) / self.Focal
        valid_idx = []
        for num in range(hand_points.shape[0]):
            if any(hand_points[num, :]):
                valid_idx.append(num)

        hand_points = hand_points[valid_idx, :]

        return hand_points

    def _rotate(self, hand_points):
        pca_mean = np.mean(hand_points, axis=0)
        nor = hand_points - pca_mean
        (_, _, coeff) = np.linalg.svd(nor, full_matrices=False)
        coeff = np.transpose(coeff)
        if coeff[1, 0] < 0:
            coeff[:, 0] = -coeff[:, 0]
        if coeff[2, 2] < 0:
            coeff[:, 2] = -coeff[:, 2]
        coeff[:, 1] = np.cross(coeff[:, 2], coeff[:, 0])
        sio.savemat('./compare/coeff.mat', {'coeff': coeff})
        hand_points_rotate = np.dot(hand_points, coeff)

        return hand_points_rotate

    def _sampling(self, hand_points_rotate):

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
        x_min_max = [np.min(hand_points_rotate[:, 0]),
                     np.max(hand_points_rotate[:, 0])]
        y_min_max = [np.min(hand_points_rotate[:, 1]),
                     np.max(hand_points_rotate[:, 1])]
        z_min_max = [np.min(hand_points_rotate[:, 2]),
                     np.max(hand_points_rotate[:, 2])]
        scale = 1.2
        bb3d_x_len = scale*(x_min_max[1]-x_min_max[0])
        bb3d_y_len = scale*(y_min_max[1]-y_min_max[0])
        bb3d_z_len = scale*(z_min_max[1]-z_min_max[0])
        max_bb3d_len = bb3d_x_len

        hand_points_normalized_sampled = hand_points_rotate_sampled / max_bb3d_len
        sio.savemat('./compare/hand_points_normalized_sampled-4.mat',
                    {'hand_points_normalized_sampled': hand_points_normalized_sampled})
        if hand_points_rotate.shape[0] < self.SAMPLE_NUM:
            offset = np.mean(hand_points_rotate, axis=0)/max_bb3d_len
        else:
            offset = np.mean(hand_points_normalized_sampled, axis=0)
        hand_points_normalized_sampled = hand_points_normalized_sampled - offset

        return hand_points_normalized_sampled

    def _farthest_point_sampling_fast(self, point_cloud, sample_num):

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

    def _bin_depth(self):
        "get point cloud from depth informations"
        header = self.data['header']
        depth = self.data['depth']
        valid_pixel_num = depth.size
        img_width = header[0]
        img_height = header[1]
        bb_left = header[2]
        bb_top = header[3]
        bb_right = header[4]
        bb_bottom = header[5]
        bb_height = bb_bottom - bb_top
        bb_width = bb_right - bb_left

        hand_3d = np.zeros((valid_pixel_num, 3))
        # '-' get on the right position
        hand_3d[:, 2] = depth
        depth = depth.reshape(bb_height, bb_width)
        h_matrix = np.array([i for i in range(bb_height)], dtype=np.float32)
        w_matrix = np.array([i for i in range(bb_width)], dtype=np.float32)

        for h in range(bb_height):
            hand_3d[(h * bb_width):((h + 1) * bb_width), 0] = np.multiply(
                (w_matrix + bb_left - (img_width / 2)), depth[h, :]) / self.fFocal_msra

        for w in range(bb_width):
            idx = [(hi * bb_width + w) for hi in range(bb_height)]
            # '-' get on the right position
            hand_3d[idx, 1] = -np.multiply(
                (h_matrix+bb_top - (img_height / 2)), depth[:, w]) / self.fFocal_msra
            # drop the useless point
        valid_idx = []
        for num in range(hand_3d.shape[0]):
            if any(hand_3d[num, :]):
                valid_idx.append(num)

        hand_points = hand_3d[valid_idx, :]

        return hand_points


def read_bin(f_name):
    "read all bin files"
    with open(f_name, 'r') as f:
        # in the bin fils, the first 6 informations are image width,
        # image height, box left, box top, box right and box bottom
        # and the rest are the depth information of the image
        header = np.fromfile(f, dtype=np.int32, count=6)
        depth = np.fromfile(f, dtype=np.float32)

    return header, depth


if __name__ == "__main__":
    file_name = 'D:/Codes/MSRA HandPoseDataset/cvpr15_MSRAHandGestureDB/P0/1/000000_depth.bin'
    header, depth = read_bin(file_name)
    data = {'depth': depth,
            'header': header}
    pp = preproces(data)
    hand_points = pp._bin_depth()
    sio.savemat('./compare/hand_points-1.mat', {'hand_points': hand_points})
    hand_points_rotate = pp._rotate(hand_points)
    sio.savemat('./compare/hand_points_rotate-2.mat',
                {'hand_points_rotate': hand_points_rotate})
    point_cloud_sampling = pp._sampling(hand_points_rotate)
    sio.savemat('./compare/point_cloud_sampling-3.mat',
                {'point_cloud_sampling': point_cloud_sampling})
    hand_points_normalized_sampled = pp._normalize(
        hand_points_rotate, point_cloud_sampling)
    sio.savemat('./compare/hand_points_normalized_sampled-5.mat',
                {'hand_points_normalized_sampled': hand_points_normalized_sampled})
    point_cloud_1, points_cloud = pp.get_point_cloud(
        hand_points_normalized_sampled)
    sio.savemat('./compare/PC-1.mat', {'point_cloud_1': point_cloud_1})
    sio.savemat('./compare/PC-2.mat', {'points_cloud': points_cloud})
    print("done!")
