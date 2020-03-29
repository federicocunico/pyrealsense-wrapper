import os
import shutil
import cv2
from PIL import Image
import imageio
from lib.realsense import RealSense

# import ffmpeg

# Not working -> non readable
# def write_video(out_file, frames, fps=30, size=(640, 480)):
#     out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
#     for i in range(len(frames)):
#         # writing to a image array
#         out.write(frames[i])
#     out.release()


# def write_video(fn, images, framerate=30, vcodec='libx264'):
# if not isinstance(images, np.ndarray):
#     images = np.asarray(images)
# n, height, width, channels = images.shape
# process = (
#     ffmpeg
#     .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
#     .output(fn, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
#     .overwrite_output()
#     .run_async(pipe_stdin=True)
# )
# for frame in images:
#     process.stdin.write(
#         frame
#         .astype(np.uint8)
#         .tobytes()
#     )
# process.stdin.close()
# process.wait()


def write_video(fn, images, fps=30, format='mp4'):
    writer = imageio.get_writer(fn, format=format, mode='I', fps=fps)
    for i in range(len(images)):
        frame = images[i]

        writer.append_data(frame)
    writer.close()


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
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # frames.append(rgb.copy())  # N.B. copy
        frames.append(rgb)

    cv2.destroyAllWindows()
    rs.close_stream()

    cwd = os.getcwd()

    # save frames
    # save_jpgs()

    # ffmpeg
    write_video(os.path.join(cwd, 'output.mp4'), frames)

    # opencv
    # write_video(os.path.join(cwd, 'output.mp4'), frames, size=(resolution[1], resolution[1]))


if __name__ == "__main__":
    resolution = (480, 640)
    run(resolution=resolution)
