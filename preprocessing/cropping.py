import cv2
import numpy as np

point = []
cropping = False

"""
https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
"""

def clickCallback(event,x,y,flags,param):
    global point, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        point = [(x,y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        point.append((x,y))
        cropping = False

        cv2.rectangle(image, point[0], point[1], (0,255,0),2)
        cv2.imshow("image",image)


"""
https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=namedwindow#namedwindow
"""

class ImageCropper(object):
    def __init__(self, window_dims):
        self.window_dims = window_dims
        self.window_name = "Image Cropper"
        cv2.namedWindow(self.window_name, cv2.CV_WINDOW_AUTOSIZE)
        # myCallbackFunc should be a callback func that deals with click/crop
        cv2.setMouseCallback(self.window_name, clickCallback)

    def display(self, image):
        # we're assuming that image is a numpy array
        # need a loop to check waitkey
        cv2.imshow(self.window_name,image)
