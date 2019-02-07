# USAGE
# python tracker.py --video ball_tracking_example.mp4
# python tracker.py

# import the necessary packages
import pyrealsense2 as rs
import numpy as np
import cv2
import imutils
import time

ALLOWED_RADIUS = 15
WIDTH = 640
HEIGHT = 480
NEAR = 300
FAR = 500
RESIZED_WIDTH = 600
RESIZED_HEIGHT = 450
FPS = 30
THRESHOLD_HEIGHT = 215

#TODO: calibrate dynamically and calculate prespective effect
TOP_LEFT = (550, 280)
TOP_RIGHT = (180, 280)
BOTTOM_LEFT = (450, 500)
BOTTOM_RIGHT = (240, 500)

blueLower = (100, 150, 20)
blueUpper = (130, 255, 255)


def createPipline():
    Pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    Pipeline.start(config)
    return Pipeline


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
    mask = cv2.inRange(hsv, blueLower, blueUpper)
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
    cXorig = round(cXre * (WIDTH / RESIZED_WIDTH))
    cYorig = round(cYre * (HEIGHT / RESIZED_HEIGHT))
    return (cXre, cYre), (cXorig, cYorig)


def PostProcessing(Filters, DepthFrame):
    DepthFrame = Filters["ZD"].process(DepthFrame)
    DepthFrame = Filters["S"].process(DepthFrame)
    DepthFrame = Filters["T"].process(DepthFrame)
    DepthFrame = Filters["DZ"].process(DepthFrame)
    DepthFrame = Filters["H"].process(DepthFrame)
    return DepthFrame


pipeline = createPipline()
filters = createFilters()
lastPoint = None
paper = np.zeros((BOTTOM_LEFT[1] - TOP_LEFT[1], TOP_LEFT[0] - TOP_RIGHT[0], 3), np.uint8)
prespectiveFallOff = (TOP_LEFT[0] - TOP_RIGHT[0])/(BOTTOM_LEFT[0] - BOTTOM_RIGHT[0])

time.sleep(2.0)

# keep looping
while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        pipeline.stop()
        break
    elif key == ord("c"):
        paper = np.zeros((BOTTOM_LEFT[1] - TOP_LEFT[1], TOP_LEFT[0] - TOP_RIGHT[0], 3), np.uint8)

    frame, depth = Fetch(pipeline)
    depth = PostProcessing(filters, depth)
    colorized_depth = ColorizeDepth(depth)
    depth = np.asanyarray(depth.get_data())

    frameResized = imutils.resize(frame, width=RESIZED_WIDTH)
	#TODO: use object detectio instead of color detection
    cnts = Contours(frameResized)
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius >= ALLOWED_RADIUS:
            (cXr, cYr), (cX, cY) = getCenter(center, (x, y))
            Z = int(depth[cY, cX])
            dZ = min(max(0, int(Z-TOP_LEFT[1])), BOTTOM_LEFT[1])
            if cY < THRESHOLD_HEIGHT or not (TOP_LEFT[1] < Z < BOTTOM_LEFT[1]) or not((cX-TOP_RIGHT[0]) < cX < TOP_LEFT[0]):
                lastPoint = None
            else:
                distanceFactor = ((1 - dZ/paper.shape[1]) + (dZ/paper.shape[1]) * prespectiveFallOff)
                dX = round((min(max(0, int(cX-TOP_RIGHT[0])), TOP_LEFT[0]) - paper.shape[0]/2) * distanceFactor + paper.shape[0]/2)
                if lastPoint is None:
                    lastPoint = (dX, dZ)
                else:
                    cv2.line(paper, lastPoint, (dX, dZ), (255, 255, 255), 1)
                    lastPoint = (dX, dZ)


            # TODO: remove after debugging
            ###################################################################
            text = "X: " + str(cX) + ",Y: " + str(cY) + ",Z: " + str(depth[cY, cX])
            cv2.circle(frameResized, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.putText(frameResized, text, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 125, 125), 2)
            cv2.circle(frameResized, (cXr, cYr), 2, (0, 0, 255), -1)
            cv2.circle(colorized_depth, (cX, cY), 2, (0, 255, 0), -1)
            ###################################################################
    viewport = paper.copy()
    viewport = cv2.flip(viewport, 1)

    cv2.namedWindow('Frame', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Paper', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Paper', 440, 740)

    cv2.imshow("Frame", frameResized)
    cv2.imshow("Depth", colorized_depth)
    cv2.imshow("Paper", viewport)

# close all windows
cv2.destroyAllWindows()
