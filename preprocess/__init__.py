import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def preprocess_image(img):
    blurred = blur(img)
    threshold_blur = thres_hold_bitwise(blurred)
    dilate_threshold_blur = dilate_image(threshold_blur)
    # cv2.imshow('dilate_threshold_blur', dilate_threshold_blur)
    return dilate_threshold_blur