"""

This file detects any motion in the video and display the direction of the motion.

Method : Optical flow is used. It is present as an in-built function available in opencv.
       : We use pixels in the contours detected for every frame in the video as features whose motion
         needs to be detected.

Input : FASTEST CARROM MATCH - YouTube (360p).mp4(enclosed in week1 folder)

Output :  Video showing any motion represented in the form of blue arrows.

Status : Working !

"""

#required libraries are imported
import cv2
import numpy as np

#Function that detects contours in the frame passed as argument
def select_point(cap,old_points,gray_frame,frame):

    # thresholding the frame as a part of pre-processing.
    ret, thresh_frame = cv2.threshold(gray_frame, 101, 255, 0)

    # finding the contours
    modified_frame, contours, hierarchy = cv2.findContours(thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # drawing contours
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    #uncomment the line below to see contours detected in every frame
    #cv2.imshow("frame with contours",frame)

    contours = np.asarray(contours)

    #appending the points on lying on the contours to old_points
    for i in contours:
        for j in i:
            x=j[0][0]
            y=j[0][1]
            old_points=np.append(old_points,[[x,y]],axis=0)
            old_points=old_points.astype(np.float32)

    return old_points,thresh_frame

def main():

    # obtaining the object for using the video
    cap = cv2.VideoCapture('FASTEST CARROM MATCH - YouTube (360p).mp4')

    #reading a frame
    _, frame = cap.read()

    # start from a later point in time in video to skip the initial bits
    while int(cap.get(cv2.CAP_PROP_POS_FRAMES)) <600: #skipping first 600 frames
        _,frame=cap.read()

    #convert the frame into gray-scale
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    """ Lucas-kanade params initialised which are used as input parameters to the in-built function :
       calcOpticalFlowPyrLK
       Parameters description : 
       a) winSize : size of the search window at each pyramid level.
       b) maxLevel : 0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), 
                       if set to 1, two levels are used, and so on
       c) criteria :specifying the termination criteria of the iterative search algorithm (after the 
                   specified maximum number of iterations criteria(10 in this case).
                   maxCount or when the search window moves by less than criteria.epsilon(=0.03).

       Using Pyramid, large motions is estimated roughly in low resolution images. This rough estimation is 
       refined in stages with high resolution images. Doing this, not only the final estimation will be 
       accurate, also it will not be time consuming as before. For this case, 5 levels of pyramid is enough to
       capture large motion and refine it with high resolution images. 

    """
    lk_params = dict(winSize=(25,25),
                     maxLevel=5,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    """
    Uncomment this to save the result of this file as a video.
    #saving the video
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 3.0, (np.size(old_gray,1),np.size(old_gray,0)))"""

    # From here we will begin a continuous process of detection movement between consecutive frames using
    # the pixels on the contours detected in every frame as features.
    while int(cap.get(cv2.CAP_PROP_POS_FRAMES)) < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):# condition for checking whether video has ended or not.The next frame number should always be less than total numbe of frames in the video.

        # getting points of the old frame(previous frame) that are part of contours detected as features.
        #Unlike in release 1 , here in every frame points lying on the contours are captured and appended
        #to old_points.
        old_points = np.array([[0, 0]], dtype=np.float32)
        old_points,thresh_frame = select_point(cap, old_points, old_gray,frame)#function to append all pixels(considered as features) in the contours detected in the frame to old points
        old_points = np.delete(old_points, 0, 0)

        # getting the next frame and its corresponding gray-scale.
        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # For the function cv2.calcOpticalFlowPyrLK() we pass the previous frame, previous points and next
        #  frame. It returns next points along with some status numbers which has a value of 1 if next point
        #  is found, else zero. We iteratively pass these next points as previous points in next step.
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame,old_points, None, **lk_params)

        old_gray = gray_frame.copy()# the current frame becomes the old frame for further motion detection.

        #making the displacement arrows for the features(points on the contours in the frame) which shows movement.
        for i in range(0,len(old_points)):
            # initialising the starting and ending points of the arrow.
            x1=int(old_points[i][0])
            y1=int(old_points[i][1])
            x2=int(new_points[i][0])
            y2=int(new_points[i][1])
            cv2.arrowedLine(frame,(x1,y1),(x2,y2),(255,0,0),1,line_type=8,shift=0,tipLength=0.1)

        """
        uncomment this to save the file
        #saving the frame frame
        out.write(frame)"""

        #showing the frame with optical flow vectors
        cv2.imshow("Frame",frame)

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