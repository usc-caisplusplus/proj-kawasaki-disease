import numpy as np
import cv2
import os

point = ()
"""
https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
"""


def clickCallback(event, x, y, flags, param):
    global point

    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)


if __name__ == '__main__':
    dir = 'dataset/Other_Images'
    paths = os.listdir(dir)

    dir_eye = 'dataset/Other_Images/Eyes'
    dir_lip = 'dataset/Other_Images/Lips'
    dir_tongue = 'dataset/Other_Images/Tongues'
    dir_skin = 'dataset/Other_Images/Skin'

    eye_count = 0
    lip_count = 0
    tongue_count = 0
    skin_count = 0

    cv2.namedWindow('Grid', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Grid', clickCallback)

    for path in paths:
        image = cv2.imread(os.path.join(dir, path))
        if image is not None:
            h, w, c = image.shape
            # Set resolution and draw grid
            res = 32
            drawing = image.copy()
            for i in range(0, h+res, res):
                cv2.line(drawing, (i, 0), (i, w), (255, 255, 255), thickness=2)
            for j in range(0, w+res, res):
                cv2.line(drawing, (0, j), (h, j), (255, 255, 255), thickness=2)

            # Set initial mode
            mode = 'e'

            while True:
                cv2.imshow('Grid', drawing)
                ch = cv2.waitKey(1)

                if len(point) == 2:
                    x_i = point[0]//res
                    y_i = point[1]//res

                    color = None
                    if mode == 'e': color = (255, 100, 0)
                    if mode == 'l': color = (0, 0, 255)
                    if mode == 't': color = (255, 0, 255)
                    if mode == 's': color = (200, 200, 200)

                    cv2.rectangle(drawing, (x_i*res, y_i*res), (x_i*res+res, y_i*res+res), thickness=-1, color = color)


                # r res
                if ch == 114:
                    point = ()
                    res = int(input('Square size: '))
                    drawing = image.copy()
                    for i in range(0, h+res, res):
                        cv2.line(drawing, (i, 0), (i, w), (255, 255, 255), thickness=2)
                    for j in range(0, w+res, res):
                        cv2.line(drawing, (0, j), (h, j), (255, 255, 255), thickness=2)
                # e eye
                if ch == 101:
                    point = ()
                    mode = 'e'
                # l lips
                if ch == 108:
                    point = ()
                    mode = 'l'
                # t tongue
                if ch == 116:
                    point = ()
                    mode = 't'
                # s skin
                if ch == 115:
                    point = ()
                    mode = 's'
                # escape (move on)
                if ch == 27:
                    point = ()
                    break
                # enter (save)
                if ch == 13:
                    if len(point) == 2:
                        x_i = point[0] // res
                        y_i = point[1] // res

                        cropped = image[y_i*res:(y_i+1)*res, x_i*res:(x_i+1)*res]
                        cv2.imshow('Save', cv2.pyrUp(cv2.pyrUp(cropped)))
                        ch2 = cv2.waitKey(0)

                        if ch2 == 13:
                            path = None
                            if mode == 'e':
                                path = os.path.join(dir_eye, '%d.png' % eye_count)
                                eye_count += 1
                            if mode == 'l':
                                path = os.path.join(dir_lip, '%d.png' % lip_count)
                                lip_count += 1
                            if mode == 't':
                                path = os.path.join(dir_tongue, '%d.png' % tongue_count)
                                tongue_count += 1
                            if mode == 's':
                                path = os.path.join(dir_skin, '%d.png' % skin_count)
                                skin_count += 1

                            cv2.imwrite(path, cropped)
                            print('Saved to {}!'.format(path))

                        cv2.destroyWindow('Save')
