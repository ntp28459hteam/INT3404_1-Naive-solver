import cv2
import numpy as np
import matplotlib.pyplot as plt
import operator

import preprocess

def detect_contour(img):
    return None
    
    
def find_corners(img):
	_, contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
	contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
	polygon = contours[0]  # Largest image

	# Use of `operator.itemgetter` with `max` and `min` allows us to get the index of the point
	# Each point is an array of 1 coordinate, hence the [0] getter, then [0] or [1] used to get x and y respectively.

	# Bottom-right point has the largest (x + y) value
	# Top-left has point smallest (x + y) value
	# Bottom-left point has smallest (x - y) value
	# Top-right point has largest (x - y) value
	bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

	# Return an array of all 4 points using the indices
	# Each point is in its own array of one coordinate
	corners = [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]
	cv2.imshow('test', display_corners(img, corners))
	return corners
	
def display_corners(img, points, radius=5, color=(0, 255, 255)):
    # Convert image from grayscale to bgr to debug
    cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for point in points:
        img = cv2.circle(img, tuple(list(point)), radius, color, -1)
    return img

def read_image(file_path):
    img_org = cv2.imread(file_path)
    img = cv2.imread(file_path, 0)
    
    preprocessed = preprocess.preprocess_image(img)
    
    cv2.imshow('preprocessed', preprocessed)
    cv2.imshow('grayscale', img)
    new_img, ext_contours, hier = cv2.findContours(preprocessed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_img, contours, hier = cv2.findContours(preprocessed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    # Draw all of the contours on the image in 2px red lines
    
    all_contours = cv2.drawContours(preprocessed.copy(), contours, -1, (255, 0, 0), 2)
    external_only = cv2.drawContours(preprocessed.copy(), ext_contours, -1, (255, 0, 0), 2)
    
    # cv2.imshow('all_contours', all_contours)
    # cv2.imshow('external_only', external_only)
    
    # plot_many_images([all_contours, external_only], ['All Contours', 'External Only'])
    
    corners = find_corners(preprocessed)
    
    display_corners(preprocessed, corners)
    
    
    # display_points(processed, corners)
    cv2.waitKey(0)

def main():
    file_path = 'sudoku-original.jpg'
    # file_path = 'nam.jpg'
    read_image(file_path)

if __name__ == '__main__':
    main()