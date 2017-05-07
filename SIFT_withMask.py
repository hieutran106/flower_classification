import cv2
import numpy as np


def drawSIFTKeypoint(fname, mask=False):
    img = cv2.imread(fname)
    (h, w) = img.shape[:2]
    (cX, cY) = ((int)(w * 0.5), (int)(h * 0.5))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (mask):
        # construct an elliptical mask representing the center of the
        # image
        (axesX, axesY) = ((int)((w * 0.8) / 2), (int)((h * 0.8) / 2))
        ellipMask = np.zeros(img.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        sift_mask = cv2.SIFT()
        kp_mask, des_mask = sift_mask.detectAndCompute(gray,ellipMask, None)
        img_kp_mask = cv2.drawKeypoints(gray, kp_mask, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow("Mask:", ellipMask)
        cv2.imshow("Keypoint with mask:",img_kp_mask)
        cv2.waitKey(0)

    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray, None)
    img_kp = cv2.drawKeypoints(gray, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #write to file
    #cv2.imwrite('sift_keypoints.jpg', img)
    #display
    print "Number of keypoint: ",str(len(des))
    cv2.imshow("Original:", img)
    cv2.imshow("Keypoint", img_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    drawSIFTKeypoint('./test_images/image_1149.jpg',mask=True)
    #drawSIFTKeypoint('./test_images/image_1149.jpg')