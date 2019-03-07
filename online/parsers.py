import argparse


def init_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int,
                        default=1, help='input batch size')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers')
    parser.add_argument('--nepoch', type=int, default=60,
                        help='number of epochs to train for')
    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    # CUDA_VISIBLE_DEVICES=0 python train.py
    parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id')

    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='learning rate at t=0')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (SGD only)')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0005, help='weight decay (SGD only)')
    parser.add_argument('--learning_rate_decay', type=float,
                        default=1e-7, help='learning rate decay')

    parser.add_argument('--SAMPLE_NUM', type=int, default=1024,
                        help='number of sample points')
    parser.add_argument('--JOINT_NUM', type=int, default=21,
                        help='number of joints')
    parser.add_argument('--INPUT_FEATURE_NUM', type=int,
                        default=3,  help='number of input point features')
    parser.add_argument('--OUTPUT', type=int, default=63,
                        help='number of PCA components')
    parser.add_argument('--knn_K', type=int, default=64,
                        help='K for knn search')
    parser.add_argument('--sample_num_level1', type=int,
                        default=512,  help='number of first layer groups')
    parser.add_argument('--sample_num_level2', type=int,
                        default=128,  help='number of second layer groups')
    parser.add_argument('--ball_radius', type=float, default=0.015,
                        help='square of radius for ball query in level 1')
    parser.add_argument('--ball_radius2', type=float, default=0.04,
                        help='square of radius for ball query in level 2')

    parser.add_argument('--test_index', type=int, default=0,
                        help='test index for cross validation, range: 0~8')
    parser.add_argument('--save_root_dir', type=str,
                        default='results',  help='output folder')
    parser.add_argument('--model', type=str, default='',
                        help='model name for training resume')
    parser.add_argument('--optimizer', type=str, default='',
                        help='optimizer name for training resume')

    opt = parser.parse_args()

    return opt
