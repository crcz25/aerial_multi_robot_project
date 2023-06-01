import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

def main():

    window_name='color range parameter'
    cv2.namedWindow(window_name)
    # Create a black image, a window
    im = cv2.imread('Drone Image_screenshot_11.04.2023.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # im = cv2.imread('stop_sign.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    im = cv2.imread('whites.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # im = cv2.imread('colors_2.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    cb = cv2.imread('colors_2.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # Increase the size of the image
    cb = cv2.resize(cb, (0,0), fx=2, fy=2)
    # cb = cv2.imread('OIP.jpeg', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # Resize image
    # cb = cv2.resize(cb, (0,0), fx=0.5, fy=0.5)
    hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    hsv_cb = cv2.cvtColor(cb,cv2.COLOR_BGR2HSV)

    print ('lower_color = np.array([a1,a2,a3])')
    print ('upper_color = np.array([b1,b2,b3])')


    # create trackbars for color change
    cv2.createTrackbar('a1',window_name,0,180,nothing)
    cv2.createTrackbar('a2',window_name,0,255,nothing)
    cv2.createTrackbar('a3',window_name,0,255,nothing)

    cv2.createTrackbar('b1',window_name,150,180,nothing)
    cv2.createTrackbar('b2',window_name,150,255,nothing)
    cv2.createTrackbar('b3',window_name,150,255,nothing)

    while(1):
        a1 = cv2.getTrackbarPos('a1',window_name)
        a2 = cv2.getTrackbarPos('a2',window_name)
        a3 = cv2.getTrackbarPos('a3',window_name)

        b1 = cv2.getTrackbarPos('b1',window_name)
        b2 = cv2.getTrackbarPos('b2',window_name)
        b3 = cv2.getTrackbarPos('b3',window_name)

        # hsv hue sat value
        lower_color = np.array([a1,a2,a3])
        upper_color = np.array([b1,b2,b3])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        mask_cb = cv2.inRange(hsv_cb, lower_color, upper_color)
        res = cv2.bitwise_and(im, im, mask = mask)
        res_cb = cv2.bitwise_and(cb, cb, mask = mask_cb)

        cv2.imshow('mask',mask)
        cv2.imshow('res',res)
        cv2.imshow('im',im)
        # cv2.imshow('hsv',hsv)
        # cv2.imshow('res_cb',res_cb)
        cv2.imshow(window_name,res_cb)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:         # wait for ESC key to exit
            break
        elif k == ord('s'): # wait for 's' key to save and exit
            cv2.imwrite('Img_screen_mask.jpg',mask)
            cv2.imwrite('Img_screen_res.jpg',res)
            break


    cv2.destroyAllWindows()


#Run Main
if __name__ == "__main__" :
    main()