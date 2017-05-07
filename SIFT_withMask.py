import cv2
import numpy as np


def drawSIFTKeypoint(fname, mask=False,ratio=0.8):
    img = cv2.imread(fname)
    (h, w) = img.shape[:2]
    (cX, cY) = ((int)(w * 0.5), (int)(h * 0.5))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (mask):
        # construct an elliptical mask representing the center of the
        # image
        (axesX, axesY) = ((int)((w * ratio) / 2), (int)((h * ratio) / 2))
        ellipMask = np.zeros(img.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        sift_mask = cv2.SIFT()
        kp_mask, des_mask = sift_mask.detectAndCompute(gray,ellipMask, None)
        img_kp_mask = cv2.drawKeypoints(gray, kp_mask, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        print "number of features with mask: ",str(len(des_mask))," zz:",str(len(kp_mask))
        cv2.imshow("Mask:", ellipMask)
        cv2.imshow("Keypoint with mask:",img_kp_mask)
        cv2.waitKey(0)

    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray, None)
    img_kp = cv2.drawKeypoints(gray, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #display
    print "Number of keypoint: ",str(len(des))
    cv2.imshow("Original:", img)
    cv2.imshow("Keypoint", img_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testMemory():
    PRE_ALLOCATION_BUFFER = 300  # for sift
    NUMBER_OF_DESCRIPTOR = 300  # for sift
    nkeys=80*17
    array = np.zeros((nkeys * PRE_ALLOCATION_BUFFER, 128),dtype =np.int8)
    print "done"

def testContrastThreshold(fname):
    img = cv2.imread(fname)
    (h, w) = img.shape[:2]
    (cX, cY) = ((int)(w * 0.5), (int)(h * 0.5))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # construct an elliptical mask representing the center of the
    # image
    (axesX, axesY) = ((int)((w * 0.8) / 2), (int)((h * 0.8) / 2))
    ellipMask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

    for i in range(0,8):
        for j in range (0,8):

            contrastThreshold=0.06+i*0.01
            edgeThreshold = 10 -j*1
            sift_mask = cv2.SIFT(contrastThreshold=contrastThreshold,edgeThreshold=edgeThreshold)
            kp_mask, des_mask = sift_mask.detectAndCompute(gray, ellipMask, None)
            img_kp_mask = cv2.drawKeypoints(gray, kp_mask, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            print "number of features with mask: ", str(len(des_mask))
            print "contrast Threshold:",str(contrastThreshold)," edgeThreshold:",str(edgeThreshold)

            cv2.imshow("Keypoint with mask:", img_kp_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



if __name__ == '__main__':
    #drawSIFTKeypoint('./images/Buttercup/image_1169.jpg',mask=True,ratio=0.85)
    #testMemory()
    testContrastThreshold('./images/Buttercup/image_1123.jpg')