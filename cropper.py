import cv2
import numpy as np

#CHANGE CROP DESIRED SIZE HERE
CROP_SIZE = 100 # size of cropped image desired (x by x in pixels)

point = (0,0)
cropNum = 0 #for filename of cropped pictures (0.jpg)


"""
https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
"""

#callback function on mouse click, updates coordinates of where mouse is clicked
#mouse click on center of desired cropped image
def clickCallback(event,x,y,flags,param):
    global point

    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x,y)

#returns array with two tuples (left top vertex and bottom right vertex)
def getPoints(point,image):
    x = point[0]
    y = point[1]
    imageX = image.shape[1]
    imageY = image.shape[0]

    #keeps rectangle within image bounds
    leftX = max(0,x-CROP_SIZE/2)
    leftY = max(0,y-CROP_SIZE/2)
    rightX = min(imageX, x + CROP_SIZE/2)
    rightY = min(imageY, y + CROP_SIZE/2)

    #ensures that if rectangle was cut off due to image size, size of rectangle stays the same
    if leftX == 0:
        rightX = CROP_SIZE
    elif rightX == imageX:
        leftX = imageX - CROP_SIZE
    
    if leftY == 0:
        rightY = CROP_SIZE
    elif rightY == imageY:
        leftY = imageY - CROP_SIZE

    return [(int(leftX), int(leftY)), (int(rightX),int(rightY))]



"""
https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=namedwindow#namedwindow
"""

#image cropper class
class ImageCropper(object):
    def __init__(self, window_dims):
        self.window_dims = window_dims
        self.window_name = "Image Cropper"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, clickCallback)


    def display(self, image):
        global CROP_SIZE, point, cropNum
        clone = image.copy()

        #if size we want of image is too big, decrease size to min dimension of original image
        if CROP_SIZE >= min(image.shape[0], image.shape[1]):
            CROP_SIZE = min(image.shape[0], image.shape[1])

        #start at left top orientation
        point = (CROP_SIZE/2,CROP_SIZE/2)

        while True:
            cv2.imshow(self.window_name,image)
            key = cv2.waitKey(1) & 0xFF

            #calculate points for rectangle
            rect = getPoints(point,image)

            #reset image as rectangle moves around
            image = clone.copy()

            #draw rectangle on image
            cv2.rectangle(image, rect[0], rect[1], (0,255,0),2)
            cv2.imshow(self.window_name,image)
            cv2.waitKey(1) & 0xFF

            # quit cropping if user enters 'q'
            if key == ord('q'):
                break
            
            #crop when enter key is pressed
            elif key == 13:
                crop = clone[rect[0][1]:rect[1][1],rect[0][0]:rect[1][0]]
                cv2.imshow("ROI",crop)
                cropKey = cv2.waitKey(0)
                cv2.imwrite(str(cropNum) + ".jpg",crop)
                cropNum += 1 #new file name for next one

        cv2.destroyAllWindows()


#testing -- use class by initializing and calling the display function with desired image
image = cv2.imread("test.jpg")
cropper = ImageCropper(4)
cropper.display(image)
