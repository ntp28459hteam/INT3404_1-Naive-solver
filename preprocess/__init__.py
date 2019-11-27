import cv2
import numpy as np
import matplotlib.pyplot as plt

import operator

def blur(origin):
    blurred = cv2.GaussianBlur(origin.copy(), (9, 9), 0)
    # cv2.imshow('blur', blurred)
    return blurred

def thres_hold_bitwise(blurred):
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    threshold = cv2.bitwise_not(threshold, threshold)
    # cv2.imshow('threshold', threshold)
    return threshold

def dilate_image(threshold):
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    dilated = cv2.dilate(threshold, kernel)
    return dilated

def preprocess_image(img, dilate=True):
    blurred = blur(img)
    threshold_blur = thres_hold_bitwise(blurred)
    
    if dilate:
        dilate_threshold_blur = dilate_image(threshold_blur)
        # cv2.imshow('blurred', blurred)
        # cv2.imshow('threshold_blur', threshold_blur)
        # cv2.imshow('dilate_threshold_blur', dilate_threshold_blur)
        return dilate_threshold_blur
    return threshold_blur

# The following function will help the solver to fit the sudoku puzzle to a 'perfect square'
    
def distance_between(p1, p2):
	# Returns the scalar distance between two points
	a = p2[0] - p1[0]
	b = p2[1] - p1[1]
	return np.sqrt((a ** 2) + (b ** 2))


def crop_and_warp(img, crop_rect):
	# Crops and warps a rectangular section from an image into a square of similar size."""

	# Rectangle described by top left, top right, bottom right and bottom left points
	top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

	# Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
	src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

	# Get the longest side in the rectangle
	side = max([
		distance_between(bottom_right, top_right),
		distance_between(top_left, bottom_left),
		distance_between(bottom_right, bottom_left),
		distance_between(top_left, top_right)
	])

	# Describe a square with side of the calculated length, this is the new perspective we want to warp to
	dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

	# Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
	m = cv2.getPerspectiveTransform(src, dst)

	# Performs the transformation on the original image
	return cv2.warpPerspective(img, m, (int(side), int(side)))

# Crop the whole sudoku puzzle

def find_contours(preprocessed):
    new_img, ext_contours, hier = cv2.findContours(preprocessed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_img, contours, hier = cv2.findContours(preprocessed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    # Draw all of the contours on the image in 2px red lines
    
    all_contours = cv2.drawContours(preprocessed.copy(), contours, -1, (255, 0, 0), 2)
    external_only = cv2.drawContours(preprocessed.copy(), ext_contours, -1, (255, 0, 0), 2)
    
    # cv2.imshow('all_contours', all_contours)
    # cv2.imshow('external_only', external_only)
    
def find_corners(img):
	_, contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
	contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
	polygon = contours[0]  # Largest image
    # print (type(polygon))

	# Use of `operator.itemgetter` with `max` and `min` allows us to get the index of the point
	# Each point is an array of 1 coordinate, hence the [0] getter, then [0] or [1] used to get x and y respectively.

	# Bottom-right point has the largest (x + y) value
    # Top-left point has the smallest (x + y) value
	bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    # for point in polygon:
    #     x = point[0][0]
    #     y = point[0][1]
    #     bottom_right = max(bottom_right, x + y)

	corners = [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]
	# cv2.imshow('test', display_corners(img, corners))
	return corners
	
def display_corners(img, points, radius=5, color=(0, 255, 255)):
    # Convert image from grayscale to bgr to display colored corners
    if len(color) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for point in points:
        img = cv2.circle(img, tuple(list(point)), radius, color, -1)
    return img
