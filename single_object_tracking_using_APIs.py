"""

This file tracks  motion of any single object.

Method : BOOSTING,MIL,KCF,TLD,MDIANFLOW,GOTURN are used to track an object chosen at moouse-click

Input : my_video1.mp4 (enclosed in week5)

Output :  Video showing tracking of single object selected

Status : Working ! Highly Inaccurate . TLD gives the best accuracy but still not good enough.

"""

#required libraries imported
import cv2
import sys
import numpy as np

#version of opencv is detected
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

#Function to correctly order the 4 points representing the corner points of the frame whose
# perspective we need to change
def order_points(pts):
    #ordering coordinates such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # top-right point will have the smallest difference,  whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

#Function for changing the view of the frame to a bird's view.
def four_point_transform(frame,pts):

    #getting consistent order for points
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # width of the new image= maximum distance between bottom-right and bottom-left
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # height of the new image = maximum distance between the top-right and bottom-right
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def main():

    # Set up tracker.
    # Instead of MIL, you can also use
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[3]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()

    # Read video
    video = cv2.VideoCapture(
        "C:\Users\MEHAR CHATURVEDI\PycharmProjects\Object_Tracking\object_tracking_OF\week5\my_video1.mp4")

    # initiating the process of changing the perspective of the frame
    # these coordinates are manually selected to get the 4 corners of the carom board.(do not tweak!)
    tl = (340, 50)  # top left
    tr = (900, 50)  # top right
    br = (840, 530)  # bottom right
    bl = (360, 530)
    corners = np.array([tl, tr, br, bl], dtype="float32")

    # Exit if video not opened.
    if not video.isOpened():
        print "Could not open video"
        sys.exit()

    # Read first frame.
    ok, frame = video.read()

    ##########################################################################################################################

    #preprocessing required for this video
    frame = cv2.flip(frame, +1)
    frame = cv2.resize(frame, (np.shape(frame)[1] / 2, np.shape(frame)[0] / 2))

    # apply the four point transform to obtain a "birds eye view" of
    # the image
    warped = four_point_transform(frame, corners)

    # resizing the warped frame
    frame= cv2.resize(warped, (np.shape(frame)[1], np.shape(frame)[0]))

    ###########################################################################

    if not ok:
        print 'Cannot read video file'
        sys.exit()

    # Define an initial bounding box
    # Uncomment the line below to select a different bounding box
    #bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while int(video.get(cv2.CAP_PROP_POS_FRAMES)) < int(video.get(cv2.CAP_PROP_FRAME_COUNT)):# condition for checking whether video has ended or not.The next frame number should always be less than total numbe of frames in the video.

        # Read a new frame
        ok, frame = video.read()

        ##########################################################################################################################

        #preprocessing required for this particular video
        frame = cv2.flip(frame, +1)
        frame = cv2.resize(frame, (np.shape(frame)[1] / 2, np.shape(frame)[0] / 2))

        # apply the four point transform to obtain a "birds eye view" of
        # the image
        warped = four_point_transform(frame, corners)

        # resizing the warped frame
        frame = cv2.resize(warped, (np.shape(frame)[1], np.shape(frame)[0]))

        ###########################################################################

        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Using the below code we are able to control the display of result according to the
        # viewer preference . Press 'p' to pause and esc key to exit.
        key = cv2.waitKey(1)
        pause = False

        # if user wishes to stop program pressesc key
        if key == 27:
            break

        if key == 112:  # 'p' has been pressed. this will pause/resume the code.
            pause = not pause
            if (pause is True):
                print("Code is paused. Press 'p' to resume..")
                while (pause is True):
                    # stay in this loop until
                    key = cv2.waitKey(0) & 0xff
                    if key == 112:
                        pause = False
                        print("Resume code..!!")
                        break

    # release the object and destroy all windows
    cv2.destroyAllWindows()
    video.release()

# program starts from here
if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()