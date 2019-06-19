import imutils
from src.Globals import constants
import cv2
from fpdf import FPDF
from PIL import Image
from PIL import ImageOps
import time
import glob


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


def save_jpg(pdf_folder, paper):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite("output/image/image_{}.png".format(timestr), paper)
    im = Image.open("output/image/image_{}.png".format(timestr))
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
