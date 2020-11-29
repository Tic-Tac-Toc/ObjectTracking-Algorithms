# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 18:19:51 2020

@author: Brice
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

roi_defined = False


def get_displacement(center, point):
    return (center[0] - point[0], center[1] - point[1])


def dx(img):

    kernel = np.array([[-1, 0, 1]])
    img3 = cv2.filter2D(img, -1, kernel)

    return img3


def dy(img):

    kernel = np.transpose(np.array([[-1, 0, 1]]))
    img3 = cv2.filter2D(img, -1, kernel)

    return img3


def get_argmax(H):
    return np.unravel_index(np.argmax(H, axis=None), H.shape)


def compute_norm(x, y):
    return(np.sqrt(x**2+y**2))


def define_ROI(event, x, y, flags, param):
    global r, c, w, h, roi_defined
    # if the left mouse button was clicked,
    # record the starting ROI coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        r, c = x, y
        roi_defined = False
    # if the left mouse button was released,
    # record the ROI coordinates and dimensions
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        h = abs(r2-r)
        w = abs(c2-c)
        r = min(r, r2)
        c = min(c, c2)
        roi_defined = True


cap = cv2.VideoCapture('Sequences/Antoine_Mug.mp4')

# take first frame of the video
ret, frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("First image", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the ROI is defined, draw it!
    if (roi_defined):
        # draw a green rectangle around the region of interest
        cv2.rectangle(frame, (r, c), (r+h, c+w), (0, 255, 0), 2)
    # else reset the image...
    else:
        frame = clone.copy()
    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        break

track_window = (r, c, h, w)
cpt = 1

# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

thetas = {}
rtablecompute = False
while(1):
    ret, frame = cap.read()
    if ret == True:
        imgup = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dilate_kernel = np.ones((3,3))
        dxIm = cv2.dilate(dx(imgup), dilate_kernel)
        dyIm = cv2.dilate(dy(imgup), dilate_kernel)

        # cv2.imshow('xo',dxIm)
        # cv2.imshow('yo',dyIm)

        magn = np.zeros(imgup.shape)
        orientations = np.zeros(imgup.shape)
        for y in range(imgup.shape[0]):
            for x in range(imgup.shape[1]):
                magn[y][x] = compute_norm(dxIm[y][x], dyIm[y][x])
                orientations[y][x] = np.arctan(dxIm[y][x]/dyIm[y][x])

        threshold = 0.2 * np.amax(magn)

        # Arrows = []
        # for i in range(magn.shape[0]):
        #     for j in range(magn.shape[1]):
        #         mgn = magn[i][j]
        #         if mgn > threshold:
        #             orientation = orientations[i][j]
        #             Arrows.append([(j, i), (  j + int((mgn/5) * np.sin(orientation)), i + int((mgn/5) * np.cos(orientation))  )])
        #             continue
        #         else :
        #             frame[i][j] = [0, 0, 255]

        # for arrow in Arrows:
        #     frame = cv2.arrowedLine(frame, arrow[0], arrow[1], (0, 255, 0), 1)

        wellfound = False
        last_track_window = track_window
        last_thetas = thetas
        iter = 0
        while not wellfound:
            iter += 1
            if iter > 5:
                thetas = last_thetas
                break
            if not rtablecompute:
                center = (int(track_window[0] + track_window[2] / 2),
                          int(track_window[1] + track_window[3] / 2))
                thetas = {}
                for y in range(track_window[2]):
                    for x in range(track_window[3]):
                        point = (track_window[0] + x, track_window[1] + y)
                        if point[0] < 0 or point[1] < 0 or point[1] >= magn.shape[0] or point[0] >= magn.shape[1]:
                            continue
                        mgn = magn[point[1]][point[0]]
                        if mgn > threshold:
                            orientation = orientations[point[1]][point[0]]
                            displacement_vector = get_displacement(
                                center, point)
                            if orientation not in thetas:
                                thetas[orientation] = []
                            thetas[orientation].append(displacement_vector)

            H = np.zeros(imgup.shape)
            for y in range(magn.shape[0]):
                for x in range(magn.shape[1]):
                    orientation = orientations[y][x]
                    mgn = magn[y][x]
                    if mgn <= threshold:
                        continue
                    if orientation in thetas:
                        for vtj in thetas[orientation]:
                            new_x, new_y = x + vtj[1], y + vtj[0]
                            if new_x < 0 or new_y < 0 or new_y >= magn.shape[0] or new_x >= magn.shape[1]:
                                continue
                            H[new_y][new_x] += 1

            
            ret, track_window = cv2.meanShift(H, track_window, term_crit)
            print("MS : " + str(ret))

            # most_probable_point = (int(track_window[0] + track_window[2] / 2), int(track_window[1] + track_window[3] / 2))

            # if most_probable_point[1] < r or most_probable_point[1] > r + h:
            #     rtablecompute = False
            #     continue
            # if most_probable_point[0] < c or most_probable_point[0] > c + w:
            #     rtablecompute = False
            #     continue

            wellfound = True

        print("Iter : " + str(iter))

        if iter > 5:
            track_window = last_track_window
              
        r, c, h, w = track_window
        frame_tracked = cv2.rectangle(
        frame, (r, c), (r+h, c+w), (255, 0, 0), 2)

        cv2.imshow('Sequence', frame_tracked)

        rtablecompute = True

        k = cv2.waitKey(60) & 0xff
        if k == 27:  # echap
            break
        elif k == ord('s'):
            cv2.imwrite('Frame_%04d.png' % cpt, frame)
        elif k == ord('c'):
            stop = True
            while stop:
                k = cv2.waitKey(60) & 0xff
                if k == ord('c'):
                    stop = False
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()
