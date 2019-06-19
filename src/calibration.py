import numpy as np
import imutils
from src.Globals import constants, utility
import cv2
from src.configs.configure import CameraHandler

class Calibrator:
    """Define a space to be able to write in"""
    def __init__(self, pipeline, profile, filters):
        self.camera_handler = CameraHandler.getInstance()
        self.DEPTH_SCALE = profile.get_device().first_depth_sensor().get_depth_scale() * 1000
        self.Edges = {}
        self.HEIGHT_THRESHOLD = 0
        for edgeStr in constants.EdgesStr:
            self.claim_edge(filters, edgeStr)
        self.HEIGHT_THRESHOLD = (self.HEIGHT_THRESHOLD / 4) - constants.MARGIN
        print("Current Threshold is {}".format(self.HEIGHT_THRESHOLD))
        self.PAPER_WIDTH = self.Edges["TopLeft"][0] - self.Edges["TopRight"][0]
        self.PAPER_HEIGHT = (self.Edges["BottomLeft"][1] - self.Edges["TopLeft"][1]) + (2 * constants.MARGIN)
        self.Near = self.Edges["TopLeft"][1] - constants.MARGIN
        self.Far = self.Edges["BottomLeft"][1] + constants.MARGIN
        self.Right = self.Edges["TopRight"][0] - constants.MARGIN
        self.Left = self.Edges["TopLeft"][0] + constants.MARGIN
        self.PrespectiveEffect = self.PAPER_WIDTH / (self.Edges["BottomLeft"][0] - self.Edges["BottomRight"][0])
        self.PAPER_WIDTH = self.PAPER_WIDTH + (2 * constants.MARGIN)
        print(self.PAPER_WIDTH, self.HEIGHT_THRESHOLD, self.PAPER_HEIGHT)

    def claim_edge(self, filters, edge):
        """Get the Edge of the space we will write in"""
        cX, cY, Z = 0, 0, 0
        while True:
            frame, depth, colorized_depth = self.camera_handler.process_frames(filters)
            frame_resized = imutils.resize(frame, width=constants.RESIZED_WIDTH)

            # TODO: use object detection instead of color detection
            cnts = utility.process_contours(frame_resized)

            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                extBot = tuple(c[c[:, :, 1].argmax()][0])
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                if radius >= constants.ALLOWED_RADIUS:
                    (cXr, cYr), (cX, cY) = utility.get_center(center, (x, y))
                    # cX, cY = extBot
                    # cX = int(round(cX * (constants.WIDTH / constants.RESIZED_WIDTH)))
                    # cY = int(round(cY * (constants.HEIGHT / constants.RESIZED_HEIGHT))) - 15
                    Z = int(depth[cY, cX] * self.DEPTH_SCALE)

                    # TODO: remove after debugging
                    ###################################################################
                    text = "X: " + str(cX) + ",Y: " + str(cY) + ",Z: " + str(Z)
                    cv2.circle(frame_resized, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.putText(frame_resized, text, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 125, 125), 2)
                    cv2.circle(frame_resized, (cXr, cYr), 2, (0, 0, 255), -1)
                    cv2.circle(colorized_depth, (cX, cY), 2, (0, 255, 0), -1)
                    ###################################################################
            else:
                cX, cY, Z = 0, 0, 0

            cv2.namedWindow('Frame', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)

            cv2.imshow("Frame", frame_resized)
            cv2.imshow("Depth", colorized_depth)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("e"):
                if cX != 0 and cY != 0 and Z != 0:
                    self.Edges[edge] = (cX, Z)
                    self.HEIGHT_THRESHOLD = self.HEIGHT_THRESHOLD + cY
                    cv2.destroyAllWindows()
                    return
