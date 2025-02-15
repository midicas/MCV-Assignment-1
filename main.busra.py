import cv2

coordinates = []
def mouseClickEventHandler(event, x, y, flags, params):
    if (event == cv2.EVENT_LBUTTONUP):
        print(x, ' ', y) 
        if len(coordinates) < 4:
            coordinates.append((x,y))
            cv2.circle(img, (x,y),5,(0, 255, 255), 2) 
            cv2.imshow("image",img)


if (__name__ == "__main__"):
    img = cv2.imread("images/Original_lena512.jpg",1)
    cv2.imshow("image",img)

    cv2.setMouseCallback("image",mouseClickEventHandler)
    cv2.waitKey(0) 
  
    # close the window 
    cv2.destroyAllWindows() 

def testImage(testImg, cameraMatrix, distCoeffs):

# Initialize    # Ik weet niet zeker of dit stukje erbij moet, want zoiets heb jij al in jouw stukje
nRows =  8
nCols = 5
termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30,0.001)
realWorldPoints = np.zeros((nRows*nCols,3), np.float32)
realWorldPoints[:,:2] = np.mgrid[0:nRows,o:nCols].T.reshape(-1,2)

# 3D axes with the origin at the center of the world coordinates and a polygon on the test image
# Color of the polygon changes with the position and orientation of the center of the top plane,    relative to the camera
axis = np.float32([[100, 0, 0],  # X-axis (red)
                       [0, 100, 0],  # Y-axis (green)
                       [0, 0, -100]]) # Z-axis (blue)		# arbitrary world coordinates

# 3D axes of the cube
cubeCorners = np.float32([[0, 0, 0], [25, 0, 0], [25, 25, 0], [0, 25, 0],
                       [0, 0, -25], [25, 0, -25], [25, 25, -25], [0, 25, -25]])

# Find corners of the chessboard
corners = detectCorners(img)
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
cubepts, _ = cv2.projectPoints(cube, rvecs, tvecs, cameraMatrix, distCoeffs)

# Convert 2D points to integer values
imgpts = np.round(imgpts).astype(int).reshape(-1, 2)
cubepts = np.round(cubepts).astype(int).reshape(-1, 2)

# Draw XYZ-axes
origin = tuple(cubepts[0])  
cv2.line(testImg, origin, tuple(imgpts[0]), (0, 0, 255), 3)  # X-axis (red)
cv2.line(testImg, origin, tuple(imgpts[1]), (0, 255, 0), 3)  # Y-axis (green)
cv2.line(testImg, origin, tuple(imgpts[2]), (255, 0, 0), 3)  # Z-axis (blue)

# Draw the polygon
edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # Base
             (4, 5), (5, 6), (6, 7), (7, 4),  # Top
             (0, 4), (1, 5), (2, 6), (3, 7)]  # Connections
for edge in edges:
        cv2.line(testImg, tuple(cubepts[edge[0]]), tuple(cubepts[edge[1]]), (255, 255, 255), 2)

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
s = max(0, min(255, int(255 * (1 – angle / 45))))

# Hue based on position of X
h = int((tvecs[0][0] + 1000) % 180)  # H between 0 and 179

# Change HSV to BGR and draw the polygon
color = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
cv2.fillConvexPoly(testImg, top_plane, color)

# Show result
cv2.imshow("3D", testImg)
cv2.waitKey(0)
        