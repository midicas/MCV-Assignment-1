import cv2
import numpy as np
import os
import sys

PATTERN_SIZE = (8, 5)

SQUARE_SIZE = 25

def click_corners(event, x, y, flags, param):
    points = param
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))

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
    bottomPoints = sorted(bottomPoints)
    
    points = topPoints + bottomPoints
    return np.array(points, dtype=np.float32)

def interpolate_grid_corners(corners):
    
    rows, cols = PATTERN_SIZE
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

def detectCorners(img, show_result=False):
    
    passed, corners = cv2.findChessboardCorners(img, PATTERN_SIZE, None)
    
    if passed:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    else:        
        outerCorners = manual_corner_selection(img)
        corners = interpolate_grid_corners(outerCorners)

    
    if show_result:
        clone = img.copy()
        cv2.drawChessboardCorners(clone, PATTERN_SIZE, corners, True)
        cv2.imshow("Detected corners", clone)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return corners
def calibrationRun(folder):
    images = [f for f in os.listdir(folder)]
    
    realWorldPoints = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1],3), np.float32)
    realWorldPoints [:,:2] = np.mgrid[0:PATTERN_SIZE[0],0:PATTERN_SIZE[1]].T.reshape(-1,2) * SQUARE_SIZE
    
    realWorldCollection = []
    imagePoints = []
    for fname in images:
        print(f"Processing Image: {fname}")
        path = os.path.join(folder,fname)
        img = cv2.imread(path)
        corners = detectCorners(img,False)

        realWorldCollection.append(realWorldPoints)
        imagePoints.append(corners)
    
    _, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(
    realWorldCollection, imagePoints, img.shape[1::-1], None, None
)
    print("Camera Matrix:\n", cameraMatrix)
    np.savez(f"{folder}.npz", 
            camera_matrix=cameraMatrix, 
            dist_coeffs=distCoeffs)

    return cameraMatrix,distCoeffs

def testImage(img,cameraMatrix,distCoeffs):
    return



if __name__ == "__main__":
    folder_path = sys.argv[1]
    cameraMatrix,distCoeffs = calibrationRun(folder_path)
    testImg = cv2.imread("test_image.jpeg")
    testImage(testImg,cameraMatrix,distCoeffs)


    