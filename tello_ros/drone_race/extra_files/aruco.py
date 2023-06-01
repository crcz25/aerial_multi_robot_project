import cv2
import numpy as np

image = cv2.imread('9.jpg')
image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

# print(f"corners: {corners}")
# print(f"ids: {ids}")
# print(f"rejected: {rejected}")

cv2.aruco.drawDetectedMarkers(image, rejected, borderColor=(100, 0, 240))
centers = []
# verify *at least* one ArUco marker was detected
if len(corners) > 0:
    # flatten the ArUco IDs list
    ids = ids.flatten()
    # loop over the detected ArUCo corners
    for (markerCorner, markerID) in zip(corners, ids):
        # extract the marker corners (which are always returned in
        # top-left, top-right, bottom-right, and bottom-left order)
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        # convert each of the (x, y)-coordinate pairs to integers
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        # draw the bounding box of the ArUCo detection
        cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
        # compute and draw the center (x, y)-coordinates of the ArUco
        # marker
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
        centers.append((cX, cY))
        # draw the ArUco marker ID on the image
        cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
        print("[INFO] ArUco marker ID: {}".format(markerID))

        # Generate the axis of the ArUco marker
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorner, 0.05, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        print(f"rvec: {rvec}")
        print(f"tvec: {tvec}")
        print(f"markerPoints: {markerPoints}")
        cv2.drawFrameAxes(image, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), np.array([0.0, 0.0, 0.0, 0.0, 0.0]), rvec, tvec, 0.1)

        # Get the pose of the ArUco marker
        rmat, _ = cv2.Rodrigues(rvec)
        pose, _ = cv2.composeRT(rmat, tvec, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), np.array([0.0, 0.0, 0.0]))
        print(f"pose: {pose}")

# verify *at least* one ArUco marker was rejected
if len(rejected) > 0:
    for i, rejectedMarker in enumerate(rejected):
        # extract the marker corners (which are always returned in
        # top-left, top-right, bottom-right, and bottom-left order)
        corners = rejectedMarker.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        # convert each of the (x, y)-coordinate pairs to integers
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        # draw the bounding box of the ArUCo detection
        cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
        # compute and draw the center (x, y)-coordinates of the ArUco
        # marker
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
        centers.append((cX, cY))
        # draw the ArUco marker ID on the image
        cv2.putText(image, "NA",(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
        print("[INFO] ArUco marker ID: {}".format("NA"))


# Get only the pose (directions) and test

# Find the center of the gate
if len(centers) > 1:
    # Calculate the center
    cX = 0
    for center in centers:
        cX += center[0]
    cX = int(cX / len(centers))
    cY = 0
    for center in centers:
        cY += center[1]
    cY = int(cY / len(centers))
    cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
    cv2.putText(image, "Center",(cX, cY - 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)

cv2.imshow("Image window", image)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()