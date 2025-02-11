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

