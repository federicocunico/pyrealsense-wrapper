import cv2
import pyrealsense2 as rs
import numpy as np
from PIL import Image


class RealSense:
    def __init__(self, resolution, fps=30, clipping_distance=1):
        """
        Initialize the realsense camera. Set the resolution, the framerate and the clipping distance
        :param resolution: tuple of (height, width) wanted from the camera
        :param fps: frame rate (default: 30)
        :param clipping_distance: clipping distance (default: 1)
        """
        h, w = resolution
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.profile = None
        self.config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.depth_scale = None
        self.align = None
        self.clipping_distance = None
        self.clipping_distance_in_meters = clipping_distance

    def open_stream(self):
        """
        Opens the stream and saves the profile
        :return:
        """
        self.profile = self.pipeline.start(self.config)

    def close_stream(self):
        """
        Closes the stream and cleans the variables
        :return:
        """
        self.pipeline.stop()
        self.profile = None
        self.align = None
        self.clipping_distance = None
        self.depth_scale = None

    def read_frames(self, raw=False):
        """
        Read Depth and RGB frames from camera as ndarray.
        If raw is set returns the frames directly in the format of realsense camera
        :param raw: if True returns the frames directly in the format of realsense camera
        :return:
        """
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if raw:
            return color_frame, depth_frame
        else:
            return self._to_numpy(color_frame), self._to_numpy(depth_frame)

    def get_depth(self, raw=False):
        """
        Gets the depth frame from camera stream.
        If raw is set returns the frame directly in the format of realsense camera
        :param raw: if True returns the frame directly in the format of realsense camera
        :return:
        """
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if raw:
            return depth_frame
        else:
            return self._to_numpy(depth_frame)

    def depth_to_display(self, depth_numpy_frame, colormap=cv2.COLORMAP_JET):
        """
        Turn a depth frame in numpy format to a colormap (default JET) in order to perform visualization.
        :param depth_numpy_frame: the depth frame input as ndarray (1-channel uint16)
        :param colormap: the requested colormap
        :return:
        """
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_numpy_frame, alpha=0.03), colormap)
        return depth_colormap

    def get_rgb(self, raw=False):
        """
        Gets the RGB frame from camera stream as ndarray.
        If raw is set returns the frame directly in the format of realsense camera
        :param raw: if True returns the frame directly in the format of realsense camera
        :return:
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if raw:
            return color_frame
        else:
            return self._to_numpy(color_frame)

    def _to_numpy(self, rs_frame):
        """
        Converts a frame from realsense format to numpy
        :param rs_frame: the input frame from RealSense
        :return:
        """
        return np.asanyarray(rs_frame.get_data())

    def get_config(self):
        """
        Returns the config of the RealSense
        :return:
        """
        return self.config

    def get_aligned_frames(self):
        """
        Gets the frames from the stream such that Depth is Aligned with RGB
        :return:
        """
        if self.depth_scale is None:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

        if self.align is None:
            # We will be removing the background of objects more than
            #  clipping_distance_in_meters meters away
            if self.clipping_distance is None:
                self._set_clipping_distance()

            # Create an align object
            # rs.align allows us to perform alignment of depth frames to others frames
            # The "align_to" is the stream type to which we plan to align depth frames.
            align_to = rs.stream.color
            self.align = rs.align(align_to)

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        return aligned_frames

    def extract_from_frames(self, frames):
        """
        Converts frames from RealSense format to Numpy ndarray
        :param frames: frames in realsense format
        :return:
        """
        depth_frame, color_frame = frames
        return self._to_numpy(color_frame), self._to_numpy(depth_frame)

    def _set_clipping_distance(self):
        """
        Set clipping distance
        :return:
        """
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale

    def remove_clipping(self, depth_image, color_image):
        """
        Remove pixels that are clipping
        :param depth_image: Depth as numpy array
        :param color_image: RGB as numpy array
        :return:
        """
        if self.clipping_distance is None:
            self._set_clipping_distance()

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.stack([depth_image] * 3, axis=-1)  # depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), grey_color,
                              color_image)
        return bg_removed

    def save_3d_model_as_ply(self):
        self._not_implemented()
        pass

    def view_3d_model(self):
        self._not_implemented()
        pass

    @staticmethod
    def _not_implemented():
        """
        Raises a warning of not implemented.
        See: https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python/examples
        for more examples
        :return:
        """
        import warnings
        s = 'Currently not implemented. See ' \
            'https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python/examples' \
            ' for more examples.'
        warnings.warn(s, RuntimeWarning)
