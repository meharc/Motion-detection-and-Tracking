"""

This program attempts to find 8 red circles , 2 present on each corner of the carom board

Method : frame is converted from bgr to hsv space and then red color is filtered by defining the hsv value range

Input :  my_video1.mp4 (enclosed in week5)

Output: video showing the circles detected in every frame.(video not saved)

Status : working! We also get false positives since red color is also present in the middle of the board.
        Also, we are not able to differentiate between the color of red circles and queen.
"""

#required libraries imported
import cv2
import numpy as np

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

#function to find contours in the given frame
def find_contours(gray_frame):

    # thresholding with threshold value=125
    ret, thresh_frame = cv2.threshold(gray_frame, 125, 255, 0)

    # finding the contours
    modified_frame, contours, hierarchy = cv2.findContours(thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

#function to draw all contours
def draw_contours(frame,contours):

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

def main():

    # to forcefully display all the elements of the output on the terminal if output is too long to be shown.
    np.set_printoptions(threshold=np.nan)

    #getting the object to use the video
    cap = cv2.VideoCapture('C:\Users\MEHAR CHATURVEDI\PycharmProjects\Object_Tracking\object_tracking_OF\week5\my_video1.mp4')

    # Create old frame
    _, frame = cap.read()

    #initiating the process of changing the perspective of the frame
    # these coordinates are manually selected to get the 4 corners of the carom board.(do not tweak!)
    tl = (340, 50)  # top left
    tr = (900, 50)  # top right
    br = (840, 530)  # bottom right
    bl = (360, 530)
    corners = np.array([tl, tr, br, bl], dtype="float32")

    #from here onwards we take each frame ,convert it into hsv space , filter the red color and then find
    #  contours on the filtered frame
    while int(cap.get(cv2.CAP_PROP_POS_FRAMES)) < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):# condition for checking whether video has ended or not.The next frame number should always be less than total numbe of frames in the video.

        #read the frame
        _, frame = cap.read()

        #preprocessing required for this specific video
        frame = cv2.flip(frame, +1)
        frame = cv2.resize(frame, (np.shape(frame)[1] / 2, np.shape(frame)[0] / 2))

        # start from a later point in time in video to skip the initial useless frames
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) > 720:  #first 720 frames skipped

            # apply the four point tranform to obtain a "birds eye view" of
            # the image
            warped = four_point_transform(frame, corners)

            # resizing the warped frame
            resized_frame = cv2.resize(warped, (np.shape(frame)[1], np.shape(frame)[0]))

            #converting to hsv space from bgr
            hsv_frame=cv2.cvtColor(resized_frame,cv2.COLOR_BGR2HSV)

            # defining the hsv value range and filtering the hsv frame for detecting red color (do not tweak the value!)
            lower_red_hue=cv2.inRange(hsv_frame,(0,50,50),(1,255,255))
            upper_red_hue=cv2.inRange(hsv_frame,(165,50,50),(179,255,255))

            # Calculate the weighted sum of lower_red_hue and upper_red_hue to cover both the results.
            red_hue_frame=cv2.addWeighted(lower_red_hue,1.0,upper_red_hue,1.0,0.0)

            #smoothening the frame by applying gaussian blur
            deblurred_frame=cv2.GaussianBlur(red_hue_frame,(5,5),0)

            # finding contours
            contours = find_contours(deblurred_frame) #function to find contours in deblurred_frame

            # draw contours
            draw_contours(resized_frame, contours) #function to draw contours

            cimg=resized_frame.copy()

            # loop over the contours
            for c in contours:
                # compute the center of the contour

                M = cv2.moments(c)

                if (int(M["m10"]) != 0 and int(M["m00"]) != 0 and int(M["m01"]) != 0):
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # draw the  center of the shape on the image
                    cv2.circle(cimg, (cX,cY), 4, (255, 255, 255), -1)

            # showing the frame
            cv2.imshow("Frame",cimg)

            # Using the below code we are able to control the display of result according to the
            # viewer preference . Press 'p' to pause and esc key to exit.
            key = cv2.waitKey(1)
            pause = False

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

    # releasing the video-object and destroying all the windows.
    cap.release()
    cv2.destroyAllWindows()

#calling the main function. Program starts from here.
if __name__ == '__main__':
    main()