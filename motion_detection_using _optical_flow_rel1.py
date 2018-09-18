"""

This file detects any motion in the video and display the direction of the motion.

Method : Optical flow is used. It is present as an in-built function available in opencv.
       : We use all pixels in the video as features whose motion needs to be detected.

Input : 7 useful Carrom shot techniques - YouTube (360p).mp4 (enclosed in week1 folder)

Output :  Video showing any motion represented in the form of blue arrows.

Status : Working !

"""

#required libraries are imported
import cv2
import numpy as np

# this function appends coordinates of all pixels in the frame passed as argument to old points.
def select_point(cap,old_points):
    for x in range(1,int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),5):
        for y in range(1,int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),5):

            old_points=np.append(old_points,[[x,y]] ,axis=0)
            old_points=old_points.astype(np.float32)

    return old_points

def main():

    # obtaining the object for using the video
    cap = cv2.VideoCapture('7 useful Carrom shot techniques - YouTube (360p).mp4')

    #reading a frame
    _,frame=cap.read()

    # start from a later point in time in video to skip the initial useless frames
    while int(cap.get(cv2.CAP_PROP_POS_FRAMES)) <600: #skipping first 600 frames
        _,frame=cap.read()

    # Create old frame and convert it into gray scale.
    _, frame = cap.read()
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
    accurate, also it will not be time consuming as before. For this case, 4 levels of pyramid is enough to
    capture large motion and refine it with high resolution images. 
                
    """
    lk_params = dict(winSize=(25,25),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    #getting points of the old frame as features
    old_points=np.array([[0,0]],dtype=np.float32)
    old_points = select_point(cap, old_points) #function to append all pixels(considered as features) in the frame to old points
    old_points = np.delete(old_points, 0, 0)

    #From here we will begin a continuous process of detection movement between consecutive frames using
    #all the pixels as features .
    while int(cap.get(cv2.CAP_PROP_POS_FRAMES)) < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):# condition for checking whether video has ended or not.The next frame number should always be less than total numbe of frames in the video.

        # getting the next frame and its corresponding gray-scale frame.
        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # For the function cv2.calcOpticalFlowPyrLK() we pass the previous frame, previous points and next
        #  frame. It returns next points along with some status numbers which has a value of 1 if next point
        #  is found, else zero. We iteratively pass these next points as previous points in next step.
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame,old_points, None, **lk_params)

        old_gray = gray_frame.copy() # the current frame becomes the old frame for further motion detection.

        #making the displacement arrows for the features(all points on the frame) which shows movement.
        for i in range(0,len(old_points)):
            #initialising the starting and ending points of the arrow.
            x1=int(old_points[i][0])
            y1=int(old_points[i][1])
            x2=int(new_points[i][0])
            y2=int(new_points[i][1])
            cv2.arrowedLine(frame,(x1,y1),(x2,y2),(255,0,0),1,line_type=8,shift=0,tipLength=0.1)

        old_points = new_points # the new points now become the old points for further motion detection

        #showing the frame
        cv2.imshow("Frame", frame)

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