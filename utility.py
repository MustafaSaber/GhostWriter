import pyrealsense2 as rs
import numpy as np
import imutils
import constants
import cv2
import configure


def createPipline():
    configure.load()
    Pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, constants.WIDTH, constants.HEIGHT, rs.format.z16, constants.FPS)
    config.enable_stream(rs.stream.color, constants.WIDTH, constants.HEIGHT, rs.format.bgr8, constants.FPS)
    Profile = Pipeline.start(config)
    return Pipeline, Profile


def createFilters():
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 1)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    spatial.set_option(rs.option.holes_fill, 3)
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    disparity_to_depth = rs.disparity_transform(False)
    depth_to_disparity = rs.disparity_transform(True)
    Filters = {"S": spatial, "T": temporal, "H": hole_filling,
               "DZ": disparity_to_depth, "ZD": depth_to_disparity}
    return Filters


def Fetch(Pipeline):
    frames = Pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    frames = align.process(frames)
    Depth_data = frames.get_depth_frame()
    RGB_frame = np.asanyarray(frames.get_color_frame().get_data())
    return RGB_frame, Depth_data


def ColorizeDepth(DepthFrame):
    colorizer = rs.colorizer()
    colorized_Depth = np.asanyarray(colorizer.colorize(DepthFrame).get_data())
    return colorized_Depth


def Contours(frameResized):
    blurred = cv2.GaussianBlur(frameResized, (11, 9), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, constants.blueLower, constants.blueUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return contours


def getCenter(centerPoint, Centroid):
    (cXre, cYre) = centerPoint
    (X, Y) = Centroid
    cXre, cYre = round((cXre + X) / 2), round((cYre + Y) / 2)
    cXorig = round(cXre * (constants.WIDTH / constants.RESIZED_WIDTH))
    cYorig = round(cYre * (constants.HEIGHT / constants.RESIZED_HEIGHT))
    return (cXre, cYre), (cXorig, cYorig)


def PostProcessing(Filters, DepthFrame):
    DepthFrame = Filters["ZD"].process(DepthFrame)
    DepthFrame = Filters["S"].process(DepthFrame)
    DepthFrame = Filters["T"].process(DepthFrame)
    DepthFrame = Filters["DZ"].process(DepthFrame)
    return DepthFrame


def align(frame, depth_frame, depth_scale, threshold=1000.00):
    # (width, height,_) = frame.shape
    threshold /= depth_scale
    new_depth = depth_frame.copy()
    newframe = frame.copy()
    new_depth[new_depth > threshold] = 0
    new_depth[new_depth > 0.0] = 1

    new_depth = np.dstack((new_depth, new_depth, new_depth))

    newframe = np.multiply(newframe, new_depth.real, dtype="uint8")
    return frame, newframe


def transformer(x, y, depthInt, scale):
    return rs.rs2_deproject_pixel_to_point(depthInt, [x, y], scale)
