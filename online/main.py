import os
import cv2
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from parsers import init_parser

# network
from net.utils import group_points
from net.network import PointNet_Plus

# data process
from preprocess import preproces

# Kinect related
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Frame, Registration
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel

try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()

print("Packet pipeline:", type(pipeline).__name__)

# Create and set logger
logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(logger)

opt = init_parser()


class Hand_estimation(object):

    def __init__(self):
        self.done = False

    def start_kinect(self):
        """
        start the Kinect and config it with only depth information output
        """

        Fn = Freenect2()
        num_devices = Fn.enumerateDevices()
        if num_devices == 0:
            print("No device connected!")
            sys.exit(1)

        serial = Fn.getDeviceSerialNumber(0)
        self.device = Fn.openDevice(serial, pipeline=pipeline)

        # select the depth data only
        self.listener = SyncMultiFrameListener(FrameType.Depth)
        self.device.setIrAndDepthFrameListener(self.listener)
        self.device.start()
        IRP = device.getIrCameraParams()
        focal = IRP.fx

    def network_config(self):
        """
        define the network and config it

        Returns:
            netR (network):  the network model
            optimizer (network): the optimizer of the network
        """

        # config network
        netR = PointNet_Plus(opt)
        if opt.ngpu > 1:
            netR.netR_1 = torch.nn.DataParallel(netR.netR_1, range(opt.ngpu))
            netR.netR_2 = torch.nn.DataParallel(netR.netR_2, range(opt.ngpu))
            netR.netR_3 = torch.nn.DataParallel(netR.netR_3, range(opt.ngpu))

        netR.load_state_dict(torch.load('./netR_58.pth'))

        netR.cuda()

        optimizer = optim.Adam(
            netR.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999), eps=1e-06)

        # optimizer.load_state_dict(torch.load('./optimizer_59.pth'))

        return netR, optimizer

    def main(self):
        """
        main part of the program
        """

        # self.start_kinect()
        Fn = Freenect2()
        num_devices = Fn.enumerateDevices()
        if num_devices == 0:
            print("No device connected!")
            sys.exit(1)

        serial = Fn.getDeviceSerialNumber(0)
        device = Fn.openDevice(serial, pipeline=pipeline)

        # select the depth data only
        listener = SyncMultiFrameListener(FrameType.Depth)
        device.setIrAndDepthFrameListener(listener)
        device.start()

        # get camera focal
        IRP = device.getIrCameraParams()
        Focal = IRP.fx

        netR, optimizer = self.network_config()

        # switch to evaluate mode
        torch.cuda.synchronize()
        netR.eval()
        print("net start")

        # record videos
        fourcc = cv2.VideoWriter_fourcc('m', 'j', 'p', 'g')
        depth_video = cv2.VideoWriter(
            'Depth.avi', fourcc, 15, (512, 424))
        seg_video = cv2.VideoWriter('Seg.avi', fourcc, 15, (512, 424), 0)

        try:
            os.mkdir('./results')
            os.mkdir('./results/dp')
            os.mkdir('./results/pc')
            os.mkdir('./results/out')
            os.mkdir('./results/off')
            os.mkdir('./results/max')
        except:
            pass

        # Get depth stream and estimate the joints location
        # count = 0
        while not self.done:

            # get depth data: 424*512 size
            frames = listener.waitForNewFrame()
            depth = frames["depth"]
            depth = depth.asarray().clip(0, 4080)   # 16*255=4080

            # convert depth to gray and RGB image
            gray_image = np.uint8(depth / 16.)
            depth_image = np.empty(
                [gray_image.shape[0], gray_image.shape[1], 3], dtype=np.uint8)
            depth_image[:, :, 0] = gray_image
            depth_image[:, :, 1] = gray_image
            depth_image[:, :, 2] = gray_image

            hand_contour = self.find_contour(gray_image)
            darks = np.zeros((424, 512), dtype=np.uint8)

            exist = True
            try:
                hand_area = cv2.contourArea(hand_contour[0])
                if hand_area < 700 or hand_area > 5000:
                    exist = False
            except:
                exist = False

            if exist:
                seg_depth, p_max, p_min = self.segmentation(
                    depth, hand_contour, darks)
                # np.save('./results/dp/depth%d.npy' % count, seg_depth)
                # show_image = self.cover
                pre = preproces(seg_depth, Focal)
                point_clouds, max_bb3d_len, offset, location = pre.run()
                # np.save('./results/pc/points_cloud%d.npy' % count, point_clouds)
                # joints = self.estimate(netR, optimizer, point_clouds)
                point_cloud = np.empty(
                    [1, opt.SAMPLE_NUM, opt.INPUT_FEATURE_NUM], dtype=np.float32)
                point_cloud[0, :, :] = point_clouds
                point_cloud = torch.from_numpy(point_cloud)
                point_cloud = point_cloud.cuda()
                inputs_level1, inputs_level1_center = group_points(
                    point_cloud, opt)
                inputs_level1 = Variable(inputs_level1, requires_grad=False)
                inputs_level1_center = Variable(
                    inputs_level1_center, requires_grad=False)

                # compute output
                optimizer.zero_grad()
                estimation = netR(inputs_level1, inputs_level1_center)
                torch.cuda.synchronize()

                # get estimation and save it
                joints = estimation.data.cpu()

                # np.save('./results/out/results%d.npy' % count, joints)
                # np.save('./results/off/offset%d.npy' % count, offset)
                # np.save('./results/max/max_bb3d_len%d.npy' % count, max_bb3d_len)
                # count += 1
                joint = joints.view(-1, 3).numpy()

                # location rematch
                joint = joint + offset

                p_middel = (p_max+p_min)/2

                for i in range(len(joint)):
                    joint[i, 0] = (joint[i, 0] * Focal) /\
                        joint[i, 2]+p_middel[0]
                    joint[i, 1] = (joint[i, 1] * Focal) /\
                        joint[i, 2]+p_middel[1]

                # print('focal:', Focal, '\nmax_bb:', max_bb3d_len)
                show_image = self.draw_joints(
                    depth_image, joint.astype(np.int32))
                cover = self.cover
            else:
                show_image = depth_image
                cover = darks

            cv2.imshow('cover', cover)
            cv2.imshow("results", show_image)

            seg_video.write(cover)
            depth_video.write(show_image)

            listener.release(frames)

            key = cv2.waitKey(delay=1)

            if key == ord('q'):
                self._done = True

        seg_video.release()
        depth_video.release()
        device.stop()
        device.close()

        sys.exit()
        # self.close_kinect()

    def find_contour(self, gray_image):
        """
        find the contour of hands

        Args:
            depth (ndarray): the original depth information

        Returns:
            hand_contour (list): the contour points list of hands
        """
        # get the hand part
        #filt_img = np.uint8(depth.copy() / 16.)
        filt_img = gray_image
        filt_img[filt_img == 0] = 255
        self.threshold = np.min(filt_img)
        filt_img[filt_img > (self.threshold + 10)] = 0

        # turn the gray image to binary image
        (_, bin_image) = cv2.threshold(filt_img,
                                       self.threshold, 255, cv2.THRESH_BINARY)

        # find contours and select the one/two with largest Area which stand for the hands
        (contours, _) = cv2.findContours(
            bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand_contour = sorted(contours, key=cv2.contourArea,
                              reverse=True)[:1]

        return hand_contour

    def segmentation(self, depth, hand_contour, darks):
        """
        segment the hand and return the segmentation image and the mask image

        Args:
            depth (ndarray): the original depth information 
            hand_contour (list): the list of hand contour
            darks (ndarray): all dark image mask

        Returns:
            seg_depth (ndarray): the segmentation depth data of hands
        """

        p_max = np.max(hand_contour, 1)[0, 0]
        p_min = np.min(hand_contour, 1)[0, 0]

        # fill the dark image with hands part
        mask = cv2.fillPoly(darks, hand_contour, 255)
        self.cover = mask.copy()
        mask = mask < (self.threshold - 1)

        depth_image = depth.copy()
        depth_image[mask] = 0
        seg_depth = depth_image[p_min[1]:p_max[1],
                                p_min[0]:p_max[0]]
        # seg_image = np.uint8(seg_image / 16.)

        return seg_depth, p_max, p_min

    def draw_joints(self, image, joints):
        """
        draw the joints in the depth image

        Args:
            image (ndarray): the gray image of depth
            joints (ndarray): the joints of hand

        Returns:
            image (ndarray): the final image for showing
        """

        # draw the points of joints
        # for i in range(joints.shape[0]):
        #     cv2.circle(image, tuple(joints[i, :2]), 5, (0, 0, 255), -1)

        # draw the lines between joints, 20 lines in total
        # for the joints, the order is: wrist(1 point), thumb(5 points), index finger(5 points)
        # middel finger(5 points), ring finger(5 points), little finger(5 points)
        color = ((160, 32, 240), (255, 0, 0),
                 (255, 165, 0), (0, 255, 0), (205, 135, 63))

        for j in range(5):
            cv2.line(image, tuple(joints[0, :2]), tuple(
                joints[1+4*j, :2]), tuple(color[j]), 2)
            for k in range(3):
                cv2.line(image, tuple(joints[1+4*j+k, :2]),
                         tuple(joints[2+4*j+k, :2]),
                         tuple(color[j]), 2)

        return image

    def close_kinect(self):
        """
        close the Kinect after using
        """

        self.device.stop()
        self.device.close()

        sys.exit()


if __name__ == "__main__":
    he = Hand_estimation()
    he.main()
