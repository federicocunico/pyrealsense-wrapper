import os
import shutil
import cv2
from PIL import Image
from lib.realsense import RealSense


def write_video(frames, out_file, fps=30, size=(640, 480)):
    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frames)):
        # writing to a image array
        out.write(frames[i])
    out.release()


def save_jpgs():
    cwd = os.getcwd()
    out_folder = os.path.join(cwd, 'current_result')
    shutil.rmtree(out_folder, ignore_errors=True)
    os.makedirs(out_folder)
    for i in range(len(frames)):
        Image.fromarray(frames[i]).save(os.path.join(out_folder, f'{i}.jpg'))


def run(resolution):
    rs = RealSense(resolution)

    rs.open_stream()
    frames = []
    while True:
        rgb, depth = rs.read_frames()
        # cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        # cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow('rgb', rgb)
        if cv2.waitKey(1) == ord('q'):
            break
        # frames.append(rgb.copy())  # N.B. copy
        frames.append(rgb)

    cv2.destroyAllWindows()
    rs.close_stream()

    cwd = os.getcwd()
    # write_video(frames, os.path.join(cwd, 'output.avi'), size=(resolution[1], resolution[1]))

    save_jpgs()


if __name__ == "__main__":
    resolution = (480, 640)
    run(resolution=resolution)
