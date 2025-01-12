# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:42:12 2020

@author: Aastha P Joshi
"""

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, grayimage, threshold=75):
    # threshold the image to get the foreground which is the hand
    thresholded = cv2.threshold(grayimage, threshold, 255, cv2.THRESH_BINARY)[1]
    print("Original image shape - " + str(image.shape))
    print("Gray image shape - " + str(grayimage.shape))

    # show the thresholded image
    cv2.imshow("Thesholded", thresholded)

    # get the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # analyze the contours
        print("Number of Contours found = " + str(len(cnts))) 
        cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
        cv2.imshow('All Contours', image) 
        
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        cv2.drawContours(image, segmented, -1, (0, 255, 0), 3)
        cv2.imshow('Max Contour', image) 
        
        return (thresholded, segmented)
print("Type of Contour: " + str(type(segmented)))
print("Contour shape: " + str(segmented.shape))
print("First 5 points in contour: " + str(segmented[:5]))
# find the convex hull of the segmented hand region
chull = cv2.convexHull(segmented)

print("Type of Convex hull: " + str(type(chull)))
print("Length of Convex hull: " + str(len(chull)))
print("Shape of Convex hull: " + str(chull.shape))

cv2.drawContours(image, [chull], -1, (0, 255, 0), 2)
cv2.imshow("Convex Hull", image)
print(chull[:,:,1])
print(chull[:,:,1].argmin())
print(chull[chull[:,:,1].argmin()])
print(chull[chull[:,:,1].argmin()][0])
print(tuple(chull[chull[:,:,1].argmin()][0]))
# find the most extreme points in the convex hull
extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

print("Extreme Top : " + str(extreme_top))
print("Extreme Bottom : " + str(extreme_bottom))
print("Extreme Left : " + str(extreme_left))
print("Extreme Right : " + str(extreme_right))

cv2.drawContours(image, [chull], -1, (0, 255, 0), 2)
cv2.circle(image, extreme_top, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_bottom, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_left, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_right, radius=5, color=(0,0,255), thickness=5)
cv2.imshow("Extreme Points in Convex Hull", image)
# find the center of the palm
cX = int((extreme_left[0] + extreme_right[0]) / 2)
cY = int((extreme_top[1] + extreme_bottom[1]) / 2)
print("Center point : " + str(tuple((cX,cY))))

cv2.drawContours(image, [chull], -1, (0, 255, 0), 2)
cv2.circle(image, (cX, cY), radius=5, color=(255,0,0), thickness=5)
cv2.circle(image, extreme_top, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_bottom, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_left, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_right, radius=5, color=(0,0,255), thickness=5)
cv2.imshow("Extreme Points in Convex Hull", image)
# find the maximum euclidean distance between the center of the palm
# and the most extreme points of the convex hull
distances = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
max_distance = distances[distances.argmax()]

# calculate the radius of the circle with 80% of the max euclidean distance obtained
radius = int(0.8 * max_distance)

# find the circumference of the circle
circumference = (2 * np.pi * radius)

print("Euclidean Distances : " + str(distances))
print("Max Euclidean Distance : " + str(max_distance))
print("Radius : " + str(radius))
print("Circumference : " + str(circumference))
# initialize circular_roi with same shape as thresholded image
circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
print("Circular ROI shape : " + str(circular_roi.shape))
cv2.imshow("Thresholded", thresholded)

# draw the circular ROI with radius and center point of convex hull calculated above
cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
cv2.imshow("Circular ROI Circle", circular_roi)

# take bit-wise AND between thresholded hand using the circular ROI as the mask
# which gives the cuts obtained using mask on the thresholded hand image
circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
cv2.imshow("Bitwise AND", circular_roi)
# compute the contours in the circular ROI
(_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("Number of Contours found = " + str(len(cnts))) 
cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
cv2.imwrite("resources/count-contours.jpg", image)
