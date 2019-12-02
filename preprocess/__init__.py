import cv2
import numpy as np
import matplotlib.pyplot as plt

import operator

from nn import NeuralNetwork
from numpy_ringbuffer import RingBuffer

# path = './IMG_20171111_153919.jpg'
path = './IMG_20191201_181702_HDR.jpg'
# path = './sudoku-original.jpg'
# path = './sudoku.png'
original_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def blur(origin):
    blurred = cv2.GaussianBlur(origin.copy(), (9, 9), 0)
    cv2.imshow('blur', blurred)
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
    # img = cv2.equalizeHist(img, img)
    blurred = blur(img)
    threshold_blur = thres_hold_bitwise(blurred)
    
    if dilate:
        dilate_threshold_blur = dilate_image(threshold_blur)
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
    return (cv2.warpPerspective(img, m, (int(side), int(side))), m)

# Crop the whole sudoku puzzle

def find_contours(preprocessed):
    new_img, ext_contours, hier = cv2.findContours(preprocessed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_img, contours, hier = cv2.findContours(preprocessed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    # Draw all of the contours on the image in 2px red lines
    
    all_contours = cv2.drawContours(preprocessed.copy(), contours, -1, (255, 0, 0), 2)
    external_only = cv2.drawContours(preprocessed.copy(), ext_contours, -1, (255, 0, 0), 2)
    
    # cv2.imshow('all_contours', all_contours)
    # cv2.imshow('external_only', external_only)
    # cv2.imwrite('all_contours.jpg', all_contours)
    # cv2.imwrite('external_only.jpg', external_only)
    
def find_corners(img):
    _, contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
    polygon = contours[0]  # Largest image
    # print (type(polygon))
    corners_image = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    external_only = cv2.drawContours(corners_image, contours, -1, (255, 0, 0), 2)
    cv2.imwrite('./debug/external_only.jpg', external_only)

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
    cv2.imwrite('./debug/corners.jpg', display_corners(img, corners, radius=int(img.shape[1] / 80) ))
    return corners
    
def display_corners(img, points, radius=25, color=(0, 255, 255)):
    # Convert image from grayscale to bgr to display colored corners
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for point in points:
        img = cv2.circle(img, tuple(list(point)), radius, color, -1)
    return img


def cut_from_rect(img, rect):
    """Cuts a rectangle from an image using the top left and bottom right points."""
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


def scale_and_centre(img, size, margin=0, background=0):
    """Scales and centres an image onto a new background square."""
    h, w = img.shape[:2]

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad,
                             r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))


def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
    """
    Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
    connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
    """
    img = inp_img.copy()  # Copy the image, leaving the original untouched
    height, width = img.shape[:2]

    max_area = 0
    seed_point = (None, None)

    if scan_tl is None:
        scan_tl = [0, 0]

    if scan_br is None:
        scan_br = [width, height]

    # Loop through the image
    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            # Only operate on light or white squares
            # Note that .item() appears to take input as y, x
            if img.item(y, x) == 255 and x < width and y < height:
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                    max_area = area[0]
                    seed_point = (x, y)

    # Colour everything grey (compensates for features outside of our middle scanning range
    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(img, None, (x, y), 64)

    # Mask that is 2 pixels bigger than the image
    mask = np.zeros((height + 2, width + 2), np.uint8)

    # Highlight the main feature
    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, mask, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 64:  # Hide anything that isn't the main feature
                cv2.floodFill(img, mask, (x, y), 0)

            # Find the bounding parameters
            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]
    return img, np.array(bbox, dtype='float32'), seed_point


def extract_digit(img, rect, size):
    """Extracts a digit (if one exists) from a Sudoku square."""

    digit = cut_from_rect(img, rect)  # Get the digit box from the whole square

    # Use fill feature finding to get the largest feature in middle of the box
    # Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    _, bbox, seed = find_largest_feature(
        digit, [margin, margin], [w - margin, h - margin])
    digit = cut_from_rect(digit, bbox)

    # Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    # Ignore any small bounding boxes
    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
        return scale_and_centre(digit, size, 4)
    else:
        return np.zeros((size, size), np.uint8)


def get_digits(img, squares, size):
    """Extracts digits from their cells and builds an array"""
    digits = []
    img = preprocess_image(img.copy(), dilate=True)
    # img = preprocess_image(img.copy(), dilate=False) # digits will be segmented if not dilate them
    for square in squares:
        digits.append(extract_digit(img, square, size))
    return digits


def infer_grid(img):
    """Infers 81 cell grid from a square image."""
    squares = []
    side = img.shape[:1]
    side = side[0] / 9
    for i in range(9):
        for j in range(9):
            p1 = (i * side, j * side)  # Top left corner of a bounding box
            # Bottom right corner of bounding box
            p2 = ((i + 1) * side, (j + 1) * side)
            squares.append((p1, p2))
    return squares

def show_digits(digits, save=None, show=False, color=255):
    """Shows a preview of what the board looks like once digits have been extracted."""
    rows = []
    # print (np.shape(digits))
    with_border = [cv2.copyMakeBorder(
        img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, color) for img in digits]
    cv2.imshow('with_border', with_border[5])
    tiles = with_border.copy()
    for i in range(9):
        row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
        # row_of_tiles = np.append(with_border[i * 9:((i + 1) * 9)], axis=1)
        # tiles.append(row_of_tiles)
        rows.append(row)
    out = np.concatenate(rows)
    # cv2.imwrite('with_border_0.jpg', rows[0])
    # cv2.imwrite('./debug/out.jpg', out) # transpose image, deprecated
    if show:
        None
        cv2.imshow('digits', out)

    if save is not None:
        cv2.imwrite('preprocessed.jpg', out)
    return out, tiles

