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
    # constrain the depth in 1~4096(255*16),
    # and reshape as the picture size
    depth = depth.clip(1, 4096).reshape(424, 512)
    original_img = np.uint8(depth.copy() / 16.)

    # using a threshold to filt the hand region
    filt_img = original_img.copy()
    filt_img[filt_img == 0] = 255
    threshod = np.min(filt_img)
    filt_img[filt_img > (threshod + 9)] = 0
    # make the hand more brighter
    # filt_img[filt_img != 0] += 100

    return depth, original_img, filt_img, threshod


def creat_mask(gray_image, threshod, flags=None, nums=0, hands_num=1):
    """
    create the mask to segment hands

    Args:
        gray_image (ndarray): the gray image after filtling
        threshod (int): the hand threshold
        flags (int, optional): Defaults to None. the flag decide the save/show operate, 
            0: only show;
            1: only save;
            2: save and show;
            others: pass
        nums (int, optional): Defaults to 0. the frame order number
        hands_num (int, optional): Defaults to 1. the number of hands, can only choose from {1,2}

    Returns:
        mask(ndarray): the mask created
        mm_point(ndarray): the max/min points of the valid mask region, the order is x_max, x_min, y_max, y_min
    """

    # turn the gray image to binary image
    (_, bin_image) = cv2.threshold(gray_image, threshod, 255, cv2.THRESH_BINARY)

    # find contours and select the one/two with largest Area which stand for the hands
    (contours, _) = cv2.findContours(
        bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_contour = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # darks = bin_image.copy()
    # darks[darks != 0] = 0
    darks = np.zeros(bin_image.shape, dtype=np.uint8)

    mask = [None]*hands_num
    for i in range(hands_num):
        hand = hand_contour[i:i + 1]
        x_max = np.max(hand, axis=0)
        x_min = np.min(hand, axis=0)
        y_max = np.max(hand, axis=1)
        y_min = np.min(hand, axis=1)
        mm_point = [x_max, x_min, y_max, y_min]
        mask[i] = cv2.fillPoly(darks, hand, 255)

        # whether to show or save the image
        if flags == 0:
            save_show(mask[i], describe='mask')
        elif flags == 1:
            save_show(mask[i], save_name='./mask/%d' % nums)
        elif flags == 2:
            save_show(mask[i], describe='mask', save_name='./mask/%d' % nums)
        else:
            pass

    return mask, mm_point


def hand_segment(depth, mask, mm_point):

    seg_image = None

    return seg_image


def save_show(image, describe=None,  save_name=None):
    """
    show or/and save the image

    Args:
        image (ndarray): image to show
        describe (string, optional): Defaults to None. the description of image
        save_name (string, optional): Defaults to None. the path and filename to save current image
    """

    if save_name == None:
        cv2.imshow('%s' % describe, image)
        cv2.waitKey(0)
    else:
        cv2.imwrite('%s.jpg' % save_name, image)

        if describe != None:
            cv2.imshow('%s' % describe, image)
            cv2.waitKey(0)


if __name__ == "__main__":
    num = 300
    depth = np.load('./frames/%d.npy' % num)
    depth, original_img, filt_img, threshod = convt_gray(depth)
    mask, mm_point = creat_mask(filt_img, threshod, flags=0)
    mask = mask[0]
    mask = mask < threshod-1
    #new_im = cv2.copyTo(depth, mask[0])
    depth[mask] = 0
    save_show(np.uint8(depth/16.), "depth")
