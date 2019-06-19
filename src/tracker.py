# import the necessary packages
import numpy as np
import cv2
from src.Globals import constants, utility, gcv_ocr
import imutils
from src import calibration
import time
import os
from src.configs.configure import CameraHandler


class Tracker:
    """Track object movement and write according to it's movement"""
    def __init__(self):
        self.camera_handler = CameraHandler.getInstance()
        self.timestr = time.strftime("%Y%m%d_%H%M%S")
        self.pdf_folder = 'output/image/pdf{}'.format(self.timestr)
        os.mkdir(self.pdf_folder)
        self.pipeline, self.profile = self.camera_handler.pipeline, self.camera_handler.profile
        self.filters = self.camera_handler.create_filters()
        self.points = None
        self.pts = None
        self.drawn = []
        self.config = calibration.Calibrator(self.profile, self.filters)
        self.paper = np.zeros((self.config.PAPER_HEIGHT, self.config.PAPER_WIDTH, 3), np.uint8) + 255
        print(self.paper.shape)
        self.track()

    def track(self):
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.save()
                self.close()
                return
            elif key == ord("c"):
                self.paper = np.zeros((self.config.PAPER_HEIGHT, self.config.PAPER_WIDTH, 3), np.uint8) + 255
                self.drawn = []
                self.points = None

            elif key == ord("s"):
                utility.save_jpg(self.pdf_folder, self.paper)
                utility.save_pdf("pdf_{}".format(self.timestr), self.pdf_folder, 'output/pdf')
                text = gcv_ocr.detect_text(self.pdf_folder)
                gcv_ocr.write_on_file(text, self.timestr)

            elif key == ord("n"):
                utility.save_jpg(self.pdf_folder, self.paper)
                self.paper = np.zeros((self.config.PAPER_HEIGHT, self.config.PAPER_WIDTH, 3), np.uint8) + 255
                self.drawn = []
                self.points = None

            frame, depth, colorized_depth = self.camera_handler.process_frames(self.filters)

            frame_resized = imutils.resize(frame, width=constants.RESIZED_WIDTH)
            # TODO: use object detection instead of color detection

            cnts = utility.process_contours(frame_resized)

            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                # extBot = tuple(c[c[:, :, 1].argmax()][0])
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                if radius >= constants.ALLOWED_RADIUS:
                    # TODO: add screen to world x,y transformer
                    (cXr, cYr), (cX, cY) = utility.get_center(center, (x, y))
                    # cX, cY = extBot
                    # cX = int(round(cX * (constants.WIDTH / constants.RESIZED_WIDTH)))
                    # cY = int(round(cY * (constants.HEIGHT / constants.RESIZED_HEIGHT))) - 15
                    # TODO: sync color and depth frame to avoid wrong depth calculation
                    Z = int(depth[cY, cX] * self.config.DEPTH_SCALE)
                    dZ = min(max(0, int(Z - self.config.Near)), self.config.Far)
                    if cY < self.config.HEIGHT_THRESHOLD or not (self.config.Near < Z < self.config.Far) or not (
                            self.config.Right < cX < self.config.Left):
                        if self.points is not None:
                            self.drawn.append(self.points)
                        self.points = None

                    else:
                        distanceFactor = ((1 - dZ / self.paper.shape[1]) +
                                          (dZ / self.config.PAPER_HEIGHT) * self.config.PrespectiveEffect)
                        dX = round(
                            (min(max(0, int(cX - self.config.Right)), self.config.Left)))
                        # - config.PAPER_WIDTH / 2) * distanceFactor
                        # + config.PAPER_WIDTH / 2)
                        if self.points is None:
                            self.points = [[dX, dZ]]
                        else:
                            self.points.append([dX, dZ])
                            self.pts = np.asanyarray(self.points, np.int32).reshape((-1, 1, 2))
                            cv2.polylines(self.paper, [self.pts], False, (0, 0, 0), lineType=cv2.LINE_AA)

                    # TODO: remove after debugging
                    ###################################################################
                    text = "X: " + str(cX) + ",Y: " + str(cY) + ",Z: " + str(Z)
                    cv2.circle(frame_resized, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.putText(frame_resized, text, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 125, 125), 2)
                    cv2.circle(frame_resized, (cXr, cYr), 2, (0, 0, 255), -1)
                    cv2.circle(colorized_depth, (cX, cY), 2, (0, 255, 0), -1)
                    ###################################################################
            viewport = self.paper.copy()
            viewport = cv2.flip(viewport, 1)

            cv2.namedWindow('Frame', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Paper', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Paper', 480, 640)

            cv2.imshow("Frame", frame_resized)
            cv2.imshow("Depth", colorized_depth)
            cv2.imshow("Paper", viewport)

    def save(self):
        """Saves the current drawn points"""
        with open('output/SavedPoints/points_{}.txt'.format(self.timestr), 'w') as f:
            for item in self.drawn:
                f.write("%s\n".replace("[", "").replace("]", "") % item)

    def close(self):
        """Close all open windows and release the camera"""
        cv2.destroyAllWindows()
        self.pipeline.stop()
