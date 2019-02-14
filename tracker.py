# import the necessary packages
import numpy as np
import cv2
import constants
import calibration
import utility
import imutils
import time

pipeline, profile = utility.createPipline()
filters = utility.createFilters()
lastPoint = None
time.sleep(2.0)

config = calibration.Calibrator(pipeline, profile, filters)

paper = np.zeros((config.PAPER_HEIGHT, config.PAPER_WIDTH, 3), np.uint8)
print(paper.shape)

# keep looping
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        pipeline.stop()
        break
    elif key == ord("c"):
        paper = np.zeros((config.PAPER_HEIGHT, config.PAPER_WIDTH, 3), np.uint8)

    frame, depth = utility.Fetch(pipeline)
    depth = utility.PostProcessing(filters, depth)
    colorized_depth = utility.ColorizeDepth(depth)
    depth = np.asanyarray(depth.get_data())

    frameResized = imutils.resize(frame, width=constants.RESIZED_WIDTH)
    # TODO: use object detection instead of color detection
    cnts = utility.Contours(frameResized)
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius >= constants.ALLOWED_RADIUS:
            # TODO: add screen to world x,y transformer
            (cXr, cYr), (cX, cY) = utility.getCenter(center, (x, y))
            # TODO: sync color and depth frame to avoid wrong depth calculation
            Z = int(depth[cY, cX] * config.DEPTH_SCALE)
            dZ = min(max(0, int(Z - config.Near)), config.Far)
            if cY < config.HEIGHT_THRESHOLD or not (config.Near < Z < config.Far) or not (
                    config.Right < cX < config.Left):
                lastPoint = None
            else:
                distanceFactor = ((1 - dZ / paper.shape[1]) + (dZ / config.PAPER_HEIGHT) * config.PrespectiveEffect)
                dX = round((min(max(0, int(cX - config.Right)), config.Left) - config.PAPER_WIDTH / 2) * distanceFactor
                           + config.PAPER_WIDTH / 2)
                if lastPoint is None:
                    lastPoint = (dX, dZ)
                else:
                    cv2.line(paper, lastPoint, (dX, dZ), (255, 255, 255), 1)
                    lastPoint = (dX, dZ)

            # TODO: remove after debugging
            ###################################################################
            text = "X: " + str(cX) + ",Y: " + str(cY) + ",Z: " + str(Z)
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
    cv2.resizeWindow('Paper', 480, 640)

    cv2.imshow("Frame", frameResized)
    cv2.imshow("Depth", colorized_depth)
    cv2.imshow("Paper", viewport)

# close all windows
cv2.destroyAllWindows()
