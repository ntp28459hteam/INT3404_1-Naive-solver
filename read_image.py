import cv2
import numpy as np
import matplotlib.pyplot as plt

import preprocess    

def read_image(file_path):
    img_org = cv2.imread(file_path)
    img = cv2.imread(file_path, 0)
    # cv2.imshow('grayscale', img)
    
    preprocessed = preprocess.preprocess_image(img)
    # cv2.imshow('preprocessed', preprocessed)

    # return list of 4 corner points
    corners = preprocess.find_corners(preprocessed)

    cropped = preprocess.crop_and_warp(img, corners)
    # cv2.imshow('cropped', cropped)

    

    cv2.waitKey(0)

def main():
    file_path = 'sudoku-original.jpg'
    read_image(file_path)

if __name__ == '__main__':
    main()