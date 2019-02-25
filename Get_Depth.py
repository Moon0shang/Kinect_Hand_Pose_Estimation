import sys
import cv2

import pygame
import ctypes
import _ctypes

import numpy as np

from threading import Thread

from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
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

        pygame.init()

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # config the Kinect
        self.config()

    def config(self):

        self.Kinect = Freenect2()
        num_devices = self.Kinect.enumerateDevices()
        if num_devices == 0:
            print("No device connected!")
            sys.exit(1)

        serial = fn.getDeviceSerialNumber(0)
        self.device = fn.openDevice(serial, pipeline=pipeline)
        self.listener = SyncMultiFrameListener(
            FrameType.Color | FrameType.Ir | FrameType.Depth)

        # Register listeners
        self.device.setColorFrameListener(listener)
        self.device.setIrAndDepthFrameListener(listener)

        self.device.start()

        # NOTE: must be called after device.start()
        self.registration = Registration(device.getIrCameraParams(),
                                         device.getColorCameraParams())

        self.undistorted = Frame(512, 424, 4)
        self.registered = Frame(512, 424, 4)

        # Optinal parameters for registration
        # set True if you need
        self.need_bigdepth = False
        self.need_color_depth_map = False

        self.bigdepth = Frame(1920, 1082, 4) if need_bigdepth else None
        self.color_depth_map = np.zeros((424, 512),  np.int32).ravel() \
            if need_color_depth_map else None

        # Optinal parameters for registration
        # set True if you need
        self.need_bigdepth = False
        self.need_color_depth_map = False

        self.bigdepth = Frame(1920, 1082, 4) if need_bigdepth else None
        self.color_depth_map = np.zeros((424, 512),  np.int32).ravel() \
            if need_color_depth_map else None

    def draw_depth_frame(self, frame, target_surface):
        if frame is None:
            return
        f8 = frame.ravel()
        frame8bit = np.dstack((f8, f8, f8))
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
        del address
        target_surface.unlock()

    def run(self):

        while self.__over:
            # --- Main event loop
            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:  # If user clicked close
                    self._done = True  # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE:  # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'],
                                                           pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)
            frames = listener.waitForNewFrame()

            color = frames["color"]
            ir = frames["ir"]
            depth = frames["depth"]

            registration.apply(color, depth, undistorted, registered,
                               bigdepth=bigdepth,
                               color_depth_map=color_depth_map)

            self.draw_depth_frame(depth, self.target_surface, t)
            depth = None

            self._screen.blit(self._frame_surface, (0, 0))
            pygame.display.update()

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            self._clock.tick(60)

        # Close our Kinect sensor, close the window and quit.
        self.device.stop()
        self.device.close()

        pygame.quit()

        sys.exit()


if __name__ == "__main__":
    Kinect_Depth = Depth()
    Kinect_Depth.run()
