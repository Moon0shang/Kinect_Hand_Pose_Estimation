import cv2
import numpy as np


def convt_gray(depth):
    """
    convert depth information to grayscale picture, 
    and filt the hand out with threshold

    Args:
        depth (ndarray): depth information from Kinect

    Returns:
        depth matrix;
        original gray picture;
        gray picture after filter;
        hand threshold
    """
    # constrain the depth in 1~4096(255*16), and reshape as the picture size
    depth = depth.clip(1, 4096).reshape(424, 512)
    original_img = np.uint8(depth.copy() / 16.)

    # using a threshold to filt the hand region
    filt_img = original_img.copy()
    filt_img[filt_img == 0] = 255
    threshod = np.min(filt_img)
    filt_img[filt_img > (threshod + 10)] = 0
    # make the hand more brighter
    # filt_img[filt_img != 0] += 100

    return depth, original_img, filt_img, threshod


def creat_mask(gray_image, threshod, hands_num):
    """
    create the mask to segment hands

    Args:
        gray_image (ndarray): the gray image after filtling
        threshod (int): the hand threshold
        hands_num (int): the number of hands, can only be choosed from {1,2}

    Returns:
        mask(ndarray): the mask created
        mm_point(ndarray): the max/min points of the valid mask region, the order is x_max, x_min, y_max, y_min
    """

    # turn the gray image to binary image
    (_, bin_image) = cv2.threshold(gray_image, threshod, 255, cv2.THRESH_BINARY)

    # find contours and select the one/two with largest Area which stand for the hands
    (contours, _) = cv2.findContours(
        bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_contour = sorted(contours, key=cv2.contourArea,
                          reverse=True)[:hands_num]

    darks = np.zeros(bin_image.shape, dtype=np.uint8)

    mask = [None] * hands_num
    p_max = [None] * hands_num
    p_min = [None] * hands_num
    for i in range(hands_num):
        hand = hand_contour[i:i + 1]

        p_max[i] = np.max(hand, 1)[0, 0]
        p_min[i] = np.min(hand, 1)[0, 0]

        # fill the dark image with hands part
        mask[i] = cv2.fillPoly(darks, hand, 255)

    return mask, p_max, p_min


def hand_segment(depth, filt_img, threshod, hands_num=1):
    """
    the segmentation of the depth image

    Args:
        depth (ndarray): the depth information
        filt_img (ndarrray): the threshold image after filtering`
        threshod (int): the threshold of hands(raw)
        hands_num (int, optional): Defaults to 1. the number of hands, can only be choosed from {1,2}

    Returns:
        seg_image(ndarray): the image after segmentation and clip
    """

    mask, p_max, p_min = creat_mask(filt_img, threshod, hands_num=hands_num)

    seg_image = [None]*hands_num
    for i in range(hands_num):
        cover = mask[i].copy()
        cover = cover < threshod - 1
        depth_image = depth.copy()
        depth_image[cover] = 0

        cv2.imshow('depth', np.uint8(depth_image / 16.))
        cv2.waitKey(0)

        # if two hands were connected, then only segment one part
        seg_image[i] = depth_image[p_min[i][1] - 2:p_max[i][1] + 2,
                                   p_min[i][0] - 2:p_max[i][0] + 2]
        if p_max[i][0] - p_min[i][0] > 120:
            seg_image.pop(1)
            break

    return seg_image


if __name__ == "__main__":
    num = 660
    depth = np.load('./frames/%d.npy' % num)
    depth, original_img, filt_img, threshod = convt_gray(depth)
    # mask, p_max, p_min = creat_mask(filt_img, threshod, flags=0)
    # mask = mask[0]
    # mask = mask < threshod-1
    # #new_im = cv2.copyTo(depth, mask[0])
    # depth[mask] = 0
    seg_image = hand_segment(depth, filt_img, threshod, hands_num=2)
    cv2.imshow("hand1", np.uint8(seg_image[0] / 16.))
    if len(seg_image) == 2:
        cv2.imshow('hand2', np.uint8(seg_image[1] / 16.))
    cv2.waitKey(0)
