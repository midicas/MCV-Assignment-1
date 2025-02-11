import cv2
import numpy as np
import os
import sys

PATTERN_SIZE = (9, 6)

SQUARE_SIZE = 1

def click_corners(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(param) < 4:
        param.append((x, y))

def manual_corner_selection(img):
    points = []
    clone = img.copy()
    
    cv2.imshow("Pick 4 corners", clone)
    cv2.setMouseCallback("Pick 4 corners", click_corners, points)
    
    while len(points) < 4:
        cv2.waitKey(1)  
    cv2.destroyAllWindows()

    points = sorted(points, key = lambda coordinate:coordinate[1])
    topPoints = points[:2]
    topPoints = sorted(topPoints)
    bottomPoints = points[2:]
    bottomPoints = sorted(bottomPoints, key = lambda coordinate:coordinate[1])
    
    points = topPoints + bottomPoints
    print(points)
    return np.array(points, dtype=np.float32)

def interpolate_grid_corners(corners, pattern_size):
    
    rows, cols = pattern_size
    grid_corners = np.zeros((rows * cols, 1, 2), dtype=np.float32)

    colMultiplier = 1/(cols-1)
    rowMultiplier = 1/(rows-1)
    
    for row in range(rows):
        for col in range(cols):
            #x-Axis interpolations:
            topInterpolation = (1-colMultiplier*col)* corners[0] + colMultiplier*col*corners[1]
            bottomInterpolation = (1-colMultiplier*col)* corners[2] + colMultiplier*col*corners[3]

            #y-axis interpolation:
            grid_corners[row * cols + col,0] = (1-rowMultiplier*row) * topInterpolation + rowMultiplier*row*bottomInterpolation
    return grid_corners

def detectCorners(img, patternSize=(7, 7), show_result=False):
    
    passed, corners = cv2.findChessboardCorners(img, patternSize, None)

    if not passed:
        outerCorners = manual_corner_selection(img)
        corners = interpolate_grid_corners(outerCorners, patternSize)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                              criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    if show_result:
        cv2.drawChessboardCorners(img, patternSize, corners, True)
        cv2.imshow("Detected corners", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return corners
def run(folder):
    images = [f for f in os.listdir(folder)]
    
    realWorldPoints = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1],3), np.float32)
    realWorldPoints [:,:2] = np.mgrid[0:PATTERN_SIZE[0],0:PATTERN_SIZE[1]].T.reshape(-1,2) * SQUARE_SIZE
    
    realWorldCollection = []
    imagePoints = []
    for fname in images:
        path = os.path.join(images,fname)
        img = cv2.imread(path)
        corners = detectCorners(img, PATTERN_SIZE)

        realWorldCollection.append(realWorldPoints)
        imagePoints.append(corners)
    
    #Calibrate Camera
    #return camera values

    return None

    



if __name__ == "__main__":
    img = cv2.imread("images/lena.jpg",1)
    detectCorners(img,(9,6),True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #detectCorners()
