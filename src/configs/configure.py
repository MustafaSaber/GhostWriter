# First import the library
import pyrealsense2 as rs
import time
import json

from src.Globals import constants
from src.Globals.constants import DS5_product_ids
import numpy as np


class CameraHandler:

    def __init__(self):
        self.pipeline = None
        self.config = None
        self.profile = None

    def find_device_that_supports_advanced_mode(self):
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) \
                    in DS5_product_ids:
                return dev
        raise Exception("No device that supports advanced mode was found")

    def load(self):
        try:
            dev = self.find_device_that_supports_advanced_mode()
            advnc_mode = rs.rs400_advanced_mode(dev)
            print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

            # Loop until we successfully enable advanced mode
            while not advnc_mode.is_enabled():
                print("Trying to enable advanced mode...")
                advnc_mode.toggle_advanced_mode(True)
                # At this point the device will disconnect and re-connect.
                print("Sleeping for 3 seconds...")
                time.sleep(3)
                # The 'dev' object will become invalid and we need to initialize it again
                dev = self.find_device_that_supports_advanced_mode()
                advnc_mode = rs.rs400_advanced_mode(dev)
                print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

            with open('src/configs/Hand.json') as f:
                json_dict = json.loads(f.read())
            json_string = json.dumps(json_dict)
            advnc_mode.load_json(json_string)

        except Exception as e:
            print(e)
            pass

    def create_pipline(self):
        """Load IntelRealsense Camera and get depth and color frames"""
        self.load()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, constants.WIDTH, constants.HEIGHT, rs.format.z16, constants.FPS)
        self.config.enable_stream(rs.stream.color, constants.WIDTH, constants.HEIGHT, rs.format.bgr8, constants.FPS)
        self.profile = self.pipeline.start(self.config)
        return self.pipeline, self.profile

    def create_filters(self):
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 5)
        spatial.set_option(rs.option.filter_smooth_alpha, 1)
        spatial.set_option(rs.option.filter_smooth_delta, 50)
        spatial.set_option(rs.option.holes_fill, 3)
        temporal = rs.temporal_filter()
        hole_filling = rs.hole_filling_filter()
        disparity_to_depth = rs.disparity_transform(False)
        depth_to_disparity = rs.disparity_transform(True)
        filters = {"S": spatial, "T": temporal, "H": hole_filling,
                   "DZ": disparity_to_depth, "ZD": depth_to_disparity}
        return filters

    def fetch(self, pipeline):
        """To get next frame and align depth frame on RGB frame"""
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.depth)
        frames = align.process(frames)
        color_frame_preprocessing = frames.get_color_frame()
        depth_data = frames.get_depth_frame()
        # The commented lines are too remove unnecessary pixels in frames
        # depth_image = np.asanyarray(Depth_data.get_data())
        color_frame = np.asanyarray(color_frame_preprocessing.get_data())
        # grey_color = 153
        # depth_image_3D = np.dstack((depth_image, depth_image, depth_image))
        # bg_removed = np.where((depth_image_3D > constants.clipping_threshold)
        # | (depth_image_3D <= 0), grey_color, RGB_frame)
        return color_frame, depth_data

    def colorize_depth(self, depth_frame):
        """Generate the heat map"""
        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        return colorized_depth

    def post_processing(self, filters, depth_frame):
        """Ably different filters to decrease noisiness from frames"""
        depth_frame = filters["ZD"].process(depth_frame)
        depth_frame = filters["S"].process(depth_frame)
        depth_frame = filters["T"].process(depth_frame)
        depth_frame = filters["DZ"].process(depth_frame)
        return depth_frame

    def process_frames(self, filters):
        frame, depth = self.fetch(self.pipeline)
        depth = self.post_processing(filters, depth)

        colorized_depth = self.colorize_depth(depth)
        depth = np.asanyarray(depth.get_data())

        return frame, depth, colorized_depth
