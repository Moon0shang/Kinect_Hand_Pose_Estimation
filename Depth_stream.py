import sys
import cv2

import numpy as np
from threading import Thread

# from Hand_segmentation import convt_gray, hand_segment

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


class Depth(object):

    def __init__(self):

        self._done = False
        self.cover = None

    def run(self):
        """
        main function to start Kinect and read depth information from Kinect, show video and save it
        """

        Fn = Freenect2()
        num_devices = Fn.enumerateDevices()
        if num_devices == 0:
            print("No device connected!")
            sys.exit(1)

        serial = Fn.getDeviceSerialNumber(0)
        device = Fn.openDevice(serial, pipeline=pipeline)
        listener = SyncMultiFrameListener(FrameType.Depth)
        device.setIrAndDepthFrameListener(listener)
        device.start()

        fourcc = cv2.VideoWriter_fourcc('m', 'j', 'p', 'g')
        depth_video = cv2.VideoWriter(
            'Depth.avi', fourcc, 30.0, (512, 424), 0)

        i = 0
        while not self._done:

            frames = listener.waitForNewFrame()
            depth = frames["depth"]
            depth = depth.asarray().clip(0, 4096)

            hand_contour = self._find_contour(depth)
            darks = np.zeros((424, 512), dtype=np.uint8)
            if cv2.contourArea(hand_contour[0]) < 1000 or cv2.contourArea(hand_contour[0]) > 5000:
                self.cover = np.uint8(depth/16.)
            else:
                seg_depth = self._segment(depth, hand_contour, darks)

            cv2.imshow("depth", self.cover)
            depth_video.write(self.cover)

            listener.release(frames)
            i += 1
            key = cv2.waitKey(delay=1)

            if key == ord('q'):
                self._done = True

        depth_video.release()
        device.stop()
        device.close()

        sys.exit()

    def _find_contour(self, depth):
        """
        find the contour of hands

        Args:
            depth (ndarray): the original depth information

        Returns:
            hand_contour (list): the contour points list of hands
        """
        # get the hand part
        filt_img = np.uint8(depth.copy() / 16.)
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

    def _segment(self, depth, hand_contour, darks):
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
        seg_depth = depth_image[p_min[1] - 1:p_max[1] + 1,
                                p_min[0] - 1:p_max[0] + 1]
        # seg_image = np.uint8(seg_image / 16.)

        return seg_depth


if __name__ == "__main__":
    dp = Depth()
    dp.run()
