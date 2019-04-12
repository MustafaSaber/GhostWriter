import utility
import numpy as np
import imutils
import constants
import cv2
import align

class Calibrator:
    def claimEdge(self, pipeline, filters, edge):
        cX, cY, Z = 0, 0, 0
        while True:
            frame, depth = utility.Fetch(pipeline)
            depth = utility.PostProcessing(filters, depth)
            colorized_depth = utility.ColorizeDepth(depth)
            depth = np.asanyarray(depth.get_data())
            _, frame = align.align(frame, depth, self.DEPTH_SCALE,constants.THRESHOLD)

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
                    (cXr, cYr), (cX, cY) = utility.getCenter(center, (x, y))
                    Z = int(depth[cY, cX] * self.DEPTH_SCALE)

                    # TODO: remove after debugging
                    ###################################################################
                    text = "X: " + str(cX) + ",Y: " + str(cY) + ",Z: " + str(Z)
                    cv2.circle(frameResized, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.putText(frameResized, text, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 125, 125), 2)
                    cv2.circle(frameResized, (cXr, cYr), 2, (0, 0, 255), -1)
                    cv2.circle(colorized_depth, (cX, cY), 2, (0, 255, 0), -1)
                    ###################################################################
            else:
                cX, cY, Z = 0, 0, 0

            cv2.namedWindow('Frame', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)

            cv2.imshow("Frame", frameResized)
            cv2.imshow("Depth", colorized_depth)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("e"):
                if cX != 0 and cY != 0 and Z != 0:
                    self.Edges[edge] = (cX, Z)
                    self.HEIGHT_THRESHOLD = self.HEIGHT_THRESHOLD + cY
                    cv2.destroyAllWindows()
                    return

    def __init__(self, pipeline, profile, filters):
        self.DEPTH_SCALE = profile.get_device().first_depth_sensor().get_depth_scale() * 1000
        self.Edges = {}
        self.HEIGHT_THRESHOLD = 0
        for edgeStr in constants.EdgesStr:
            self.claimEdge(pipeline, filters, edgeStr)
        self.HEIGHT_THRESHOLD = (self.HEIGHT_THRESHOLD / 4) - constants.MARGIN
        self.PAPER_WIDTH = self.Edges["TopLeft"][0] - self.Edges["TopRight"][0]
        self.PAPER_HEIGHT = (self.Edges["BottomLeft"][1] - self.Edges["TopLeft"][1]) + (2 * constants.MARGIN)
        self.Near = self.Edges["TopLeft"][1] - constants.MARGIN
        self.Far = self.Edges["BottomLeft"][1] + constants.MARGIN
        self.Right = self.Edges["TopRight"][0] - constants.MARGIN
        self.Left = self.Edges["TopLeft"][0] + constants.MARGIN
        self.PrespectiveEffect = self.PAPER_WIDTH / (self.Edges["BottomLeft"][0] - self.Edges["BottomRight"][0])
        self.PAPER_WIDTH = self.PAPER_WIDTH + (2 * constants.MARGIN)
        print(self.PAPER_WIDTH, self.HEIGHT_THRESHOLD, self.PAPER_HEIGHT)
