# Kinect_Hand_Pose_Estimation
a project use kinect and CNN to estimate hand gesture

# requirement:

- runing enviroment:
  - cv2
  - numpy
  - pytorch
  - matplotlib
  - [libfreenect2](https://github.com/OpenKinect/libfreenect2)
  - [pylibfreenect2](https://github.com/r9y9/pylibfreenect2)

- hardware:
  - computer with usb3.0
  - Kinect V2
  
# runing

- online
  - run `main.py`

- offline
  - get segmenmtation depth data from `Depth_stream.py`
  - process the data by runing `preprocess.py`
  - and last the `eval.py`
    `modify the files to meet your needs`


























