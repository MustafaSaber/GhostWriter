import cv2
import numpy as np
def DrawHermiteCurve(paper, points1,
                     points2,points3,points4):
    (x1,y1) = points1
    (x2,y2) = points2
    (x3,y3) = points3
    (x4,y4) = points4
    print(points1)

    a1 = 2 * x1 + x2 - 2 * x3 + x4

    b1 = -3 * x1 - 2 * x2 + 3 * x3 - x4
    c1 = x2
    d1 = x1
    a2 = 2 * y1 + y2 - 2 * y3 + y4
    b2 = -3 * y1 - 2 * y2 + 3 * y3 - y4
    c2 = y2
    d2 = y1

    for t in np.arange(0,1,0.001):
        x = a1*(t*t*t) + b1*(t*t) + c1*(t)+d1
        y = a2*(t*t*t) + b2*(t*t) + c2*(t)+d2

        # paper[int(y),int(x)]=(0,0,0)
        cv2.line(paper, (int(x),int(y)),(int(x)+1,int(y)+1),(0, 0, 0),1)

def Draw3Curve(paper, point1,point2,point3):

    (x1,y1) = point1
    (x2,y2) = point2
    (x3,y3) = point3


    a0 = x1
    a1 = -3 * x1 + 4 * x2 - x3
    a2 = 2 * x1 - 4 * x2 + 2 * x3
    b0 = y1
    b1 = -3 * y1 + 4 * y2 - y3
    b2 = 2 * y1 - 4 * y2 + 2 * y3
    for t in np.arange(0,1,0.001):
        x = a0 + a1 * t + a2 * t * t
        y = b0 + b1 * t + b2 * t * t
        cv2.circle(paper, (int(x),int(y)),1,(0, 0, 0))


    # def main():
#     paper = np.zeros((480,640,3), np.uint8)+255
#     DrawHermiteCurve(paper,10,100,240,200,-240,400,360,-400)
#     cv2.namedWindow('Paper', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Paper', 480, 640)
#     cv2.imshow("Paper", paper)
#     cv2.waitKey()
#
# # paper = np.zeros((400,800, 3), np.uint8) + 255
# main()
