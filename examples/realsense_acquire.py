import time
import cv2
import os
import yaml
from PIL import Image
import numpy as np
from datetime import datetime
from lib.datasets.generation.realsense import RealSense
from lib.utils.config import proj_cfg as cfg
from lib.utils.extended_utils import get_files_from


def test_run(resolution=None, min_dist=None):
    """
    test with resolution and minimum distance. if none, default is used

    :param resolution:
    :param min_dist:
    :return:
    """
    resolution = resolution if resolution is not None else (480, 640)
    min_dist = min_dist if min_dist is not None else 3500  # 350 mm  # 35cm  # 0.35 m
    realsense = RealSense(resolution)
    realsense.open_stream()

    while True:
        aligned_frames = realsense.get_aligned_frames()

        rgb_np, depth_frame_np = realsense.extract_from_frames(aligned_frames)
        depth = realsense.depth_to_display(depth_frame_np)

        res = np.zeros_like(depth_frame_np)
        res[depth_frame_np < min_dist] = depth_frame_np[depth_frame_np < min_dist]

        res = realsense.depth_to_display(res)
        # res = np.stack([res]*3, axis=-1).astype(np.uint8)
        black = np.zeros_like(rgb_np).astype(np.uint8)
        top = cv2.hconcat((rgb_np, depth))
        bot = cv2.hconcat((black, res))
        out = cv2.vconcat((top, bot))
        # cv2.imshow('color', rgb)
        # cv2.imshow('original_depth', depth)
        cv2.imshow('realsense', out)

        # idea: acquisisco sempre, e metto label 'other', quando tengo premuto una key è una label
        if cv2.waitKey(1) == ord('q'):
            break

    realsense.close_stream()


def get_hand_by_distance(depth_frame, max_dist=None, OFFSET=1000):
    """
    Returns the distance where should be the hand

    :param depth_frame: a uint16 representation of the depth
    :param OFFSET: the width of the hand in 10^-4 m
    :return: the minimum depth distance of the hand
    """
    assert len(depth_frame.shape) == 2, "The depth must be 1-channel 16bit"
    minimum = np.min(depth_frame[np.nonzero(depth_frame)])
    m = minimum + OFFSET

    if max_dist is not None and m >= max_dist:
        # If setted a max_distance and the current m is more distant from camera than max_dist, return the whole depth
        return 0  # np.max(depth_frame)  # or 0 ? TODO: try

    return m


def write_text(img, s):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 450)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    r = cv2.putText(img, s,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
    return r


def record(resolution, delay_seconds=0):
    realsense = RealSense(resolution)
    realsense.open_stream()

    # Calibration
    calib_time = 2  # seconds
    lower_bound = 0  # minimum distance
    max_dist = None
    print("Wait for calibration (~{} s)".format(calib_time))
    start = time.time()
    while True:
        if time.time() - start >= calib_time:
            break
        np_depth = realsense.extract_from_frames(realsense.get_aligned_frames())[1]
        max_dist = np.max(np_depth)
        lower_bound = get_hand_by_distance(np_depth, max_dist)

    print('Max distance found: {} meters (i.e. {} cm)'.format(max_dist / 10000, max_dist / 100))

    # Acquisition
    i = 0
    last_pressed = 0
    labels_gt = {}

    # dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'acquisition')
    dst = os.path.join(cfg.DATA_DIR, 'our_hands')
    rgb_path = os.path.join(dst, 'JPEGImages')
    mask_path = os.path.join(dst, 'mask')
    visualization_mask = os.path.join(dst, 'visualization_mask')
    labels_path = os.path.join(dst, 'labels.yaml')
    classes_path = os.path.join(dst, 'classes.yaml')

    os.makedirs(dst, exist_ok=True)
    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(visualization_mask, exist_ok=True)

    print("Performing acquisition: saving in {}".format(dst))
    start = time.time()
    while True:

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == 32:  # space
            last_pressed = 0
        else:
            if key != -1:
                last_pressed = key

        class_type = get_class(last_pressed)
        label = labels[class_type]

        # Align depth and color info
        aligned_frames = realsense.get_aligned_frames()

        # Get frames as ndarray
        rgb_np, depth_frame_np = realsense.extract_from_frames(aligned_frames)
        depth = realsense.depth_to_display(depth_frame_np)

        # Create current MASK
        res = np.zeros_like(depth_frame_np)

        # Euristic for hand distance from camera
        # lower_bound = get_hand_by_distance(depth_frame_np, max_dist)

        # print('\rMinimum: {}'.format(lower_bound), end='')

        # Gets the depth info close to the camera (lower_bound)
        res[depth_frame_np < lower_bound] = depth_frame_np[depth_frame_np < lower_bound]

        res = realsense.depth_to_display(res)
        # res = np.stack([res]*3, axis=-1).astype(np.uint8)
        black = np.zeros_like(rgb_np).astype(np.uint8)
        black = write_text(black, label)

        top = cv2.hconcat((rgb_np, depth))
        bot = cv2.hconcat((black, res))
        out = cv2.vconcat((top, bot))
        # cv2.imshow('color', rgb)
        # cv2.imshow('original_depth', depth)
        cv2.imshow('realsense', out)

        end = time.time() - start
        if end < delay_seconds:
            continue
        else:
            start = time.time()

            now = datetime.now()
            print('[{}] Acquiring class {} ({})'.format(now, class_type, label))

            rgb_filename = os.path.join(rgb_path, '{}.jpg'.format(i))
            mask_filename = os.path.join(mask_path, '{}.png'.format(i))
            vis_mask_filename = os.path.join(visualization_mask, '{}.png'.format(i))

            mask_out = res.astype(np.uint8)
            mask_out = np.sum(mask_out, axis=2)  # 3channel to 1 channel
            mask_out[mask_out > 0] = 1  # all axes != 0 to 1
            mask_out[mask_out == (255, 0, 0)] = 0  # put zero the background
            mask_out[mask_out != (0, 0, 0)] = 1  # put foreground to 1
            mask_out = mask_out.sum(axis=2)  # sum on all axis to get binary
            mask_out[mask_out > 0] = 1  # normalize to 1
            mask_out = mask_out.astype(np.uint8)

            vis_mask = res.astype(np.uint8)
            vis_mask[vis_mask > 0] = 255
            # vis_mask = np.stack([vis_mask] * 3, axis=-1)  # è già 3 channel

            if label == labels[0]:
                # Azzero la maschera in salvataggio
                mask_out = np.zeros_like(mask_out)
                vis_mask = np.zeros_like(vis_mask)

            rgb_out = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb_out).save(rgb_filename)
            Image.fromarray(mask_out).save(mask_filename)
            Image.fromarray(vis_mask).save(vis_mask_filename)

            labels_gt[i] = class_type
            with open(labels_path, 'w') as outfile:
                yaml.dump(labels_gt, outfile, default_flow_style=False)
            i += 1
    with open(classes_path, 'w') as outfile:
        yaml.dump(labels, outfile, default_flow_style=False)

    split_test_train(dst, rgb_path)
    realsense.close_stream()


def split_test_train(dst, rgb_path, train_perc=0.9, old_numpy=False):
    train_txt = os.path.join(dst, 'train.txt')
    test_txt = os.path.join(dst, 'test.txt')
    val_txt = os.path.join(dst, 'val.txt')

    rgbs = get_files_from(rgb_path, fullpath=False)

    n = np.arange(rgbs.shape[0])

    train_size = int(rgbs.shape[0] * train_perc)
    val_perc = 0.07
    test_perc = 1 - train_perc

    if old_numpy:
        train_idxs = np.random.choice(n, train_size, replace=False)
        test_idxs = [x for x in n if x not in train_idxs]
        val_idxs = []
    else:
        from sklearn.model_selection import train_test_split
        train_idxs, test_idxs = train_test_split(n, test_size=test_perc)
        train_idxs, val_idxs = train_test_split(train_idxs, test_size=val_perc)

    test_idxs.sort()
    train_idxs.sort()
    val_idxs.sort()

    with open(val_txt, 'w') as outfile:
        for t in val_idxs:
            outfile.write(rgbs[t])
            outfile.write('\n')

    with open(train_txt, 'w') as outfile:
        for t in train_idxs:
            outfile.write(rgbs[t])
            outfile.write('\n')

    with open(test_txt, 'w') as outfile:
        for t in test_idxs:
            outfile.write(rgbs[t])
            outfile.write('\n')


labels = {
    0: "Other",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Close Hand",
    7: "Ok",
}


def get_class(x):
    if x == ord('1'):
        return 1
    elif x == ord('2'):
        return 2
    elif x == ord('3'):
        return 3
    elif x == ord('4'):
        return 4
    elif x == ord('5'):
        return 5
    elif x == ord('c'):  # Close Hand
        return 6
    elif x == ord('o'):  # Ok
        return 7
    else:
        return 0


if __name__ == "__main__":
    # TODO: campionare ogni x tempo

    resolution = (480, 640)
    min_dist = 3500  # 350 mm  # 35cm  # 0.35 m

    # Listener(on_press=on_press, on_release=on_release).start()
    # listener.join()
    # pass

    # test_run()
    record(resolution, delay_seconds=3)

"""
IDEA: prendo i pixel più vicini, e uso quelli come min dist + OFFSET, dove offset è la grandezza massima consentita 
del polso per fare un simbolo, per esempio 10 cm

Min viene calcolato in runtime, così da far funzionare "bene" l'acquisizione dati

"""
