import cv2
import numpy as np
import os
import sys

PATTERN_SIZE = (8, 5)

SQUARE_SIZE = 1
CUBE_SIZE = 1
AXIS_LENGTH = 3

def click_corners(event, x, y, flags, param):
    points = param
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))

def manual_corner_selection(img):
    """"Let users select 4 corner points of the checkerboard"""
    points = []
    clone = img.copy()

    cv2.imshow("Pick 4 corners", clone)
    cv2.setMouseCallback("Pick 4 corners", click_corners, points)
    #Collect points
    while len(points) < 4:
        cv2.waitKey(1)  
    cv2.destroyAllWindows()

    #Sort points such that the final list is of the form [topLeft,topRight,bottomLeft,bottomRight]
    points = sorted(points, key = lambda coordinate:coordinate[1])
    topPoints = points[:2]
    topPoints = sorted(topPoints)
    bottomPoints = points[2:]
    bottomPoints = sorted(bottomPoints)
    
    points = topPoints + bottomPoints
    return np.array(points, dtype=np.float32)

def interpolate_grid_corners(corners):
    """Use interpolation to estimate the inner corners of the checkerboard"""
    rows, cols = PATTERN_SIZE

    #account for possible 90 degree rotation
    width = np.linalg.norm(corners[1] - corners[0]) 
    height = np.linalg.norm(corners[2] - corners[0])
    
    if height < width:  
        rows, cols = cols, rows
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
    """Takes image and return the found corners of the checkerboard
    """
    #Normal call of findChessboardCorners. Will not always work
    passed, corners = cv2.findChessboardCorners(img, PATTERN_SIZE, None)
    
    #If works refine corners 
    if passed:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    #Else let user manually select corners
    else:        
        outerCorners = manual_corner_selection(img)
        corners = interpolate_grid_corners(outerCorners)

    #If wanted, show found corners.
    if show_result:
        clone = img.copy()
        cv2.drawChessboardCorners(clone, PATTERN_SIZE, corners, True)
        cv2.imshow("Detected corners", clone)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return corners
def calibrationRun(folder):
    """Function takes in a folder name that contains the images that you want to train on"""
    images = [f for f in os.listdir(folder)]
    
    #2D real-world points of checkerboard
    realWorldPoints = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1],3), np.float32)
    realWorldPoints [:,:2] = np.mgrid[0:PATTERN_SIZE[0],0:PATTERN_SIZE[1]].T.reshape(-1,2) * SQUARE_SIZE
    
    realWorldCollection = [realWorldPoints for i in images]
    imagePoints = []
    #Build collection of all corners across images
    for fname in images:
        print(f"Processing Image: {fname}")
        path = os.path.join(folder,fname)
        img = cv2.imread(path)
        corners = detectCorners(img,False)

        imagePoints.append(corners)
    
    #Use collection of corners to calibrate camera
    _, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(
    realWorldCollection, imagePoints, img.shape[1::-1], None, None
)
    print("Camera Matrix:\n", cameraMatrix)
    np.savez(f"{folder}.npz", 
            camera_matrix=cameraMatrix, 
            dist_coeffs=distCoeffs)

    return cameraMatrix,distCoeffs
def testImage(testImg, cameraMatrix, distCoeffs,):
    
# 3D axes with the origin at the center of the world coordinates and a polygon on the test image
# Color of the polygon changes with the position and orientation of the center of the top plane,    relative to the camera


    realWorldPoints = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1], 3), np.float32)
    realWorldPoints[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

# 3D axes of the cube
    axis = np.float32([[AXIS_LENGTH, 0, 0],  # X-axis (red)
                        [0, AXIS_LENGTH, 0],  # Y-axis (green)
                        [0, 0,-AXIS_LENGTH]]).reshape(-1,3) # Z-axis (blue)		# arbitrary world coordinates

    cubeCorners = np.float32([[0, 0, 0],[0, CUBE_SIZE, 0], [CUBE_SIZE, CUBE_SIZE, 0], [CUBE_SIZE, 0, 0],
                        [0, 0, -CUBE_SIZE], [0, CUBE_SIZE, -CUBE_SIZE],[CUBE_SIZE, CUBE_SIZE, -CUBE_SIZE],[CUBE_SIZE, 0, -CUBE_SIZE] ])

    # Find corners of the chessboard
    corners = detectCorners(testImg)
    if corners is None:
            print("No chessboard found")
            return
    # Calculate the position of the camera towards the chessboard
    ret, rvecs, tvecs = cv2.solvePnP(realWorldPoints, corners, cameraMatrix, distCoeffs)
    if not ret:
            print("Pose not found")
            return

    # Project the axes and cube to 2D
    imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, cameraMatrix, distCoeffs)
    cubepts, _ = cv2.projectPoints(cubeCorners, rvecs, tvecs, cameraMatrix, distCoeffs)

    
    # Convert 2D points to integer values
    imgpts = imgpts.astype(int)
    cubepts = np.int32(cubepts).reshape(-1, 2)
    # Draw XYZ-axes
    origin = tuple(corners[0].ravel().astype(int)) 
    print(cubepts)
    print(origin)
    cv2.line(testImg, origin, tuple(imgpts[0].ravel()), (0, 0, 255), 5)  # X-axis (red)
    cv2.line(testImg, origin, tuple(imgpts[1].ravel()), (0, 255, 0), 5)  # Y-axis (green)
    cv2.line(testImg, origin, tuple(imgpts[2].ravel()), (255, 0, 0), 5)  # Z-axis (blue)

    # Draw the polygon
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # Base
                (4, 5), (5, 6), (6, 7), (7, 4),  # Top
                (0, 4), (1, 5), (2, 6), (3, 7)]  # Connections
    for edge in edges:
        cv2.line(testImg, tuple(cubepts[edge[0]]), tuple(cubepts[edge[1]]), (255, 255, 255), 5)

    # Calculate the color of the top plane
    top_plane = np.array([cubepts[4], cubepts[5], cubepts[6], cubepts[7]], dtype=np.int32)

    # Distance to camera, using Z-value of the tvecs
    distance = np.linalg.norm(tvecs)
    v = max(0, min(255, int(255 * (1 - distance / 4000))))  # V between 0 and 255

    # Orientation, angle between normal and viewing direction
    z_axis = np.array([0, 0, 1])	# A unit vector along the z-axis in world coordinates, perpendicular to the chessboard
    R, _ = cv2.Rodrigues(rvecs)	# Convert rotation vector to a rotation matrix
    normal = R @ z_axis		# The normal of the top plane in camera coordinates, Z-axis will be rotated
    # The Z-component of the normal determines how straight the plane looks to the camera
    angle = np.degrees(np.arccos(np.dot(normal, [0, 0, 1])))	# [0, 0, 1] is the Z-axis of the camera coordinates, np.dot(normal, [0, 0, 1]) calculates the cosinus of the angle between the normale and the Z-axis of the camera, np.arccos() is the angle in radians and np.degrees gives the angle in degrees
    s = max(0, min(255, int(255 * (1 - angle / 45))))

    # Hue based on position of X
    h = int((tvecs[0][0] + 1000) % 180)  # H between 0 and 179

    # Change HSV to BGR and draw the polygon
    color = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
    cv2.fillConvexPoly(testImg, top_plane, color)

    # Show result
    testImg = cv2.resize(testImg, (0, 0), fx = 0.2, fy = 0.2)
    cv2.imshow("3D", testImg)
    cv2.waitKey(0)

if __name__ == "__main__":
    folder_path = sys.argv[1]
    #cameraMatrix,distCoeffs = calibrationRun(folder_path)
    calibration_data = np.load("Run1.npz")
    cameraMatrix = calibration_data["camera_matrix"]
    distCoeffs = calibration_data["dist_coeffs"]
    testImg = cv2.imread("test.jpeg")
    testImage(testImg,cameraMatrix,distCoeffs)




    
    