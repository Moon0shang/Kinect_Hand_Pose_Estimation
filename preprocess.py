import numpy as np


class preproces(object):

    def __init__(self, seg_depth):
        self.seg_depth = seg_depth
        self.Focal = None
        self.SAMPLE_NUM = 1024
        self.sample_num_l1 = 512
        self.sample_num_l2 = 128

    def __point_cloud(self):

        [width, height] = self.seg_depth.shape
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
            hand_points[idx, 1] = np.multiply(
                h_matrix - height / 2, self.seg_depth[:w]) / self.Focal

        for num in range(hand_points.shape[0]):
            if any(hand_points[num, :]):
                valid_idx.append(num)

        hand_points = hand_points[valid_idx, :]

        return hand_points

    def rotate(self, hand_points):
        pca_mean = np.mean(point_cloud, axis=0)
        nor = point_cloud - pca_mean
        (_, _, coeff) = np.linalg.svd(nor, full_matrices=False)
        coeff = np.transpose(coeff)

        hand_points_rotate = hand_points * coeff

        return hand_points_rotate

    def __sampling(self):

        hand_points = self.__point_cloud()
        point_num = self.point_num
        point_shape = hand_points.shape[0]

        if point_shape < point_num:
            rand_idx = np.arange(0, point_num, 1, dtype=np.int32)
            rand_idx[point_shape:] = np.random.randint(0, point_shape,
                                                       size=point_num - point_shape)
        else:
            rand_idx = np.random.randint(0, point_shape, size=point_num)

        point_cloud = hand_points[rand_idx, :]

        return point_cloud, rand_idx

    def normalize(self, points_cloud):
        x_min_max = [np.min(points_cloud[:, 0]), np.max(points_cloud[:, 0])]
        y_min_max = [np.min(points_cloud[:, 1]), np.max(points_cloud[:, 1])]
        z_min_max = [np.min(points_cloud[:, 2]), np.max(points_cloud[:, 2])]
        scale = 1.2
        bb3d_x_len = scale*(x_min_max(2)-x_min_max(1))
        bb3d_y_len = scale*(y_min_max(2)-y_min_max(1))
        bb3d_z_len = scale*(z_min_max(2)-z_min_max(1))
        max_bb3d_len = bb3d_x_len

        hand_points_normalized_sampled = hand_points_rotate_sampled / max_bb3d_len
        if size(hand_points, 1) < self.SAMPLE_NUM
                offset = mean(hand_points_rotate)/max_bb3d_len;
            else
                offset = mean(hand_points_normalized_sampled);
        hand_points_normalized_sampled = hand_points_normalized_sampled - offset

        return hand_points_normalized_sampled

    def _farthest_point_sampling_fast(self,point_cloud,sample_num):
        
        pc_num = point_cloud.shape[0]
        if pc_num <= self.sample_num:
            sampled_idx = np.arange(0, point_num, 1, dtype=np.int32)
            sampled_idx[point_shape:] = np.random.randint(
                0, point_shape,size=point_num - point_shape)
        else:
            sampled_idx = np.zeros([sample_num])
            sampled_idx[0] = np.random.randint(1, pc_num)
            
            cur_sample = point_cloud[sampled_idx[0],:]  
            # cur_sample = np.tile(point_cloud[sample_idx[0],:], [sample_num, 1])
            diff = point_cloud - cur_sample
            distance = np.sum(np.multiply(diff,diff),axis=1)
            
            for i in range(1, sample_num):
                sampled_idx[i] = np.argmax(distance)

                if i < sample_num:
                    valid_idx = distance > 0
                    diff = point_cloud[valid_idx,:] - point_cloud[sampled_idx[i]]
                    new_distance = np.sum(np.multiply(diff, diff), axis=1)
                    distance[valid_idx,:] = np.min([distance[valid_idx,:], new_distance], axis=1)
                    
        return np.unique(sampled_idx)


    def get_point_cloud(self,point_cloud):


        # sample level 1
        sampled_idx_l1 = self._farthest_point_sampling_fast(point_cloud,self.sample_num_l1)
        
        other_index = np.(set(np.arange(self.SAMPLE_NUM)) -set(sampled_idx_l1))
        new_idx = np.stack((sampled_idx_l1, other_idx))
        point_cloud = point_cloud[new_idx,:]

        # sample level 2
        sampled_idx_l2 = self._farthest_point_sampling_fast(
            point_cloud[:self.sample_num_l1], self.sample_num_l2)
        other_index = np.(set(np.arange(self.sample_num_l1)) -set(sampled_idx_l2))
        new_idx = np.stack((sampled_idx_l2, other_idx))
        points_cloud = point_cloud[new_idx,:]


        return points_cloud
