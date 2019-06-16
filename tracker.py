# import the necessary packages
import numpy as np
import cv2
import constants
import calibration
import utility
import imutils
import time
import align
from fpdf import FPDF
from PIL import Image, ImageTk
from PIL import ImageOps
import tkintertable as tk

def open_img(image):
    img = Image.open(image)
    #img = img.resize((250, 250), Image.ANTIALIAS)
    #img = ImageTk.PhotoImage(image)
    time.sleep(2)
    panel = tk.Label(root, image=img)
    panel.image = img
    panel.pack()


def makePdf(pdfFileName, listPages, dir = ''):
    if (dir):
        dir += "/"
    cover = Image.open(dir + listPages)
    width, height = cover.size
    pdf = FPDF(unit = "pt", format = [width, height])
    pdf.add_page()
    pdf.image(dir + listPages,0,0)
    pdf.output(dir + pdfFileName + ".pdf", "F")
    print("pdf")

root = tk.Tk()

saveButton = tk.Button(root,text ="Save",command = makePdf)
saveButton.pack()

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

    elif key == ord("s"):
        cv2.imwrite("filename.png", paper)
        im = Image.open("filename.png")
        im = ImageOps.mirror(im)
        im.save("filename2.png")
        makePdf("firstTry", "filename2.png")

    frame, depth = utility.Fetch(pipeline)
    depth = utility.PostProcessing(filters, depth)
    colorized_depth = utility.ColorizeDepth(depth)
    depth = np.asanyarray(depth.get_data())
    _,frame = align.align(frame,depth,config.DEPTH_SCALE,constants.THRESHOLD)

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

    # cv2.namedWindow('Frame', cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow('Paper', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Paper', 480, 640)


    #cv2.imshow("Frame", frameResized)
    #cv2.imshow("Depth", colorized_depth)
    #cv2.imshow("Paper", viewport)

    # label = tk.Label(root, textvariable=paper)
    # label.pack()
    cv2.imwrite("paper.png", paper)
    #img = Image.fromarray(paper, 'RGB')
    #open_img("paper.jpg")
    #open_img(img)
    btn = tk.Button(root, text='open image', command=open_img("paper.png")).pack()
    var = tk.StringVar()
    label = tk.Label(root, textvariable=var)
    var.set("Ghost Writer")
    label.pack()
    root.mainloop()

# close all windows
cv2.destroyAllWindows()
