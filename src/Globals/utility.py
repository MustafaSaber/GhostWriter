import pyrealsense2 as rs
import numpy as np
import imutils
from src.Globals import constants
import cv2
from src.configs import configure
import math
from fpdf import FPDF
from PIL import Image
from PIL import ImageOps
import time
import glob


def create_pipline():
    """Load IntelRealsense Camera and get depth and color frames"""
    configure.load()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, constants.WIDTH, constants.HEIGHT, rs.format.z16, constants.FPS)
    config.enable_stream(rs.stream.color, constants.WIDTH, constants.HEIGHT, rs.format.bgr8, constants.FPS)

    profile = pipeline.start(config)
    return pipeline, profile


def create_filters():
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


def fetch(pipeline):
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
    # bg_removed = np.where((depth_image_3D > constants.clipping_threshold) | (depth_image_3D <= 0), grey_color, RGB_frame)
    return color_frame, depth_data


def colorize_depth(depth_frame):
    """Generate the heat map"""
    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    return colorized_depth


def process_contours(frame_resized):
    """Get contours of the object detected"""
    blurred = cv2.GaussianBlur(frame_resized, (11, 9), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, constants.blueLower, constants.blueUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return contours


def get_center(center_point, centroid):
    """Get the center of the circle drawn above the object detected"""
    (cXre, cYre) = center_point
    (X, Y) = centroid
    cXre, cYre = round((cXre + X) / 2), round((cYre + Y) / 2)
    cXorig = round(cXre * (constants.WIDTH / constants.RESIZED_WIDTH))
    cYorig = round(cYre * (constants.HEIGHT / constants.RESIZED_HEIGHT))
    return (cXre, cYre), (cXorig, cYorig)


def post_processing(filters, depth_frame):
    """Ably different filters to decrease noisiness from frames"""
    depth_frame = filters["ZD"].process(depth_frame)
    depth_frame = filters["S"].process(depth_frame)
    depth_frame = filters["T"].process(depth_frame)
    depth_frame = filters["DZ"].process(depth_frame)
    return depth_frame


def save_jpg(pdf_folder, paper):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite("../output/image/image_{}.png".format(timestr), paper)
    im = Image.open("../output/image/image_{}.png".format(timestr))
    im = ImageOps.mirror(im)
    im.save(pdf_folder+'/image_{}.png'.format(timestr))


def save_pdf(pdf_filename, pages_list, directory=''):
    if directory:
        directory += "/"

    pdf = FPDF(unit="pt", format=[1000, 1000])
    for img in glob.glob(pages_list + '/*.*'):
        try:
            pdf.add_page()
            pdf.image(str(img), 0, 0)
        except Exception as e:
            print(e)

    pdf.output(directory + pdf_filename + ".pdf", "F")
    print("You saved a PDF")
