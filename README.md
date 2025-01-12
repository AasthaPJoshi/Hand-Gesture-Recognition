# Hand Gesture Recognition using Python and OpenCV
This repository contains the implementation of a Hand Gesture Recognition System. The system uses Python and OpenCV to recognize hand gestures and count the number of fingers in real-time from a video feed.

## Features
Real-time hand gesture recognition.
Finger counting using convex hull and contour extraction.
Background subtraction for accurate hand region segmentation.

## Introduction
This project demonstrates how to process live video streams to recognize hand gestures. It isolates the hand region from the background, analyzes its shape, and calculates the number of fingers extended. The system is lightweight and can be extended to gesture-based control applications.

## Technologies Used
Programming Language: Python 3.x
Libraries:
OpenCV
NumPy
Imutils
Scikit-learn

## Usage
Start the Program:
Ensure you have a working webcam.
Run the script using python main.py.
## Perform Gestures:
Place your hand in front of the webcam within the defined ROI (Region of Interest).
Observe real-time finger counting and gesture recognition.

# Project Workflow

## Hand Region Segmentation:
Background Subtraction: Removes static background.
Motion Detection and Thresholding: Highlights moving regions.
Contour Extraction: Detects the hand's contour.

## Finger Counting:
Calculate the convex hull of the hand.
Identify the center of the palm.
Define a circular Region of Interest (ROI).
Count fingers by analyzing contours within the ROI.

## Output
Real-time video feed with:
The segmented hand region highlighted.
Finger count displayed as an overlay.
