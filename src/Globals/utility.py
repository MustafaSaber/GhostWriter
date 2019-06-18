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


def createPipline():
    configure.load()
    Pipeline = rs.pipeline()
    config = rs.config()
    ##########
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
    '''To get next frame and align depth frame on RGB frame'''
    frames = Pipeline.wait_for_frames()
    align = rs.align(rs.stream.depth)
    frames = align.process(frames)
    color_frame_preprocessing = frames.get_color_frame()
    Depth_data = frames.get_depth_frame()
    # The commented lines are too remove unnecessary pixels in frames
    # depth_image = np.asanyarray(Depth_data.get_data())
    RGB_frame = np.asanyarray(color_frame_preprocessing.get_data())
    # grey_color = 153
    # depth_image_3D = np.dstack((depth_image, depth_image, depth_image))
    # bg_removed = np.where((depth_image_3D > constants.clipping_threshold) | (depth_image_3D <= 0), grey_color, RGB_frame)
    return RGB_frame, Depth_data


def ColorizeDepth(DepthFrame):
    '''Generate the heat map'''
    colorizer = rs.colorizer()
    colorized_Depth = np.asanyarray(colorizer.colorize(DepthFrame).get_data())
    return colorized_Depth


def Contours(frameResized):
    '''Ge contours of the object detected'''
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
    '''Get the center of the circle drawn above the object detected'''
    (cXre, cYre) = centerPoint
    (X, Y) = Centroid
    cXre, cYre = round((cXre + X) / 2), round((cYre + Y) / 2)
    cXorig = round(cXre * (constants.WIDTH / constants.RESIZED_WIDTH))
    cYorig = round(cYre * (constants.HEIGHT / constants.RESIZED_HEIGHT))
    return (cXre, cYre), (cXorig, cYorig)


def PostProcessing(Filters, DepthFrame):
    '''Ably different filters to decrease noisiness from frames'''
    DepthFrame = Filters["ZD"].process(DepthFrame)
    DepthFrame = Filters["S"].process(DepthFrame)
    DepthFrame = Filters["T"].process(DepthFrame)
    DepthFrame = Filters["DZ"].process(DepthFrame)
    return DepthFrame


def transformer(x, y, depthInt, scale):
    return rs.rs2_deproject_pixel_to_point(depthInt, [x, y], scale)


def saveJPG(pdf_folder, paper):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite("../document/image/image_{}.png".format(timestr), paper)
    im = Image.open("../document/image/image_{}.png".format(timestr))
    im = ImageOps.mirror(im)
    im.save(pdf_folder+'/image_{}.png'.format(timestr))


def savePGF(pdfFileName, listPages, dir=''):
    if dir:
        dir += "/"

    pdf = FPDF(unit="pt", format=[1000, 1000])
    for img in glob.glob(listPages + '/*.*'):
        try:
            var_img = cv2.imread(img)
            pdf.add_page()
            pdf.image(str(img), 0, 0)
        except Exception as e:
            print(e)

    pdf.output(dir + pdfFileName + ".pdf", "F")
    print("You saved a PDF")

