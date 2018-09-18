"""
Single object detection and tracking.

Method: Taking absolute difference between consecutive frames( in gray-scale ) for object detection.
      : Contour detection on the thresholded frame obtained from object detection for object tracking.

Input : bouncingBall.avi (enclosed in week1 folder).

Output : Video showing the track of detected object. ( Video not saved)

Status : Working !

"""

#import required libraries
import cv2

#parameter required for performing thresholding
SENSITIVITY_VALUE=20 #threshold value (do not tweak it!)

#parameter required for smoothening the thresholded frame
BLUR_SIZE=10 #(do not tweak it!)

#starting position of the object initialised.
theObject=[0,0]

# This function helps us to track the object in a particular frame by finding the contour ,forming a
# bounding rectangle around it and getting the center of the rectangle . The frame used to perform all
# these operations is the result of the applying thresholding and deblurring on the aboslute difference of
# consecutive frames.
def searchForMovement(thresholdImage,cameraFeed):

    temp=thresholdImage.copy() #make a copy of the used frame to avoid making direct changes to it.

    #finding contours ! Since , it is single object tracking, we assume the contour detected belongs to
    #the same object.
    temp1,contours,hierarchy=cv2.findContours(temp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    if len(contours)==0: #if not contour detected
        print("object undetected")

    else :
        x,y,w,h=cv2.boundingRect(contours[0]) #get the bounding rectangle around the contour

        #get the center of the bounding rectangle
        xpos=x+w/2
        ypos=y+h/2

        theObject[0],theObject[1]=xpos,ypos #update the position of the object detected

    x,y=theObject[0],theObject[1]

    #using the center of the bounding rectangle, we draw our target symbol( a circle with a cross-section)'
    # on the object detected in the cameraFeed . The position of the object is also displayed on the
    #caameraFeed.
    cv2.circle(cameraFeed,(x,y),20,(0,255,0),2)
    cv2.line(cameraFeed,(x,y),(x,y-25),(0,255,0),2)
    cv2.line(cameraFeed, (x, y), (x, y +25), (0, 255, 0), 2)
    cv2.line(cameraFeed, (x, y), (x-25, y), (0, 255, 0), 2)
    cv2.line(cameraFeed, (x, y), (x+25, y), (0, 255, 0), 2)
    cv2.putText(cameraFeed,"Tracking Object at (" + str(x)+ ","+str(y)+")",(x,y),1,1,(255,0,0),2)

def mains():

    # variables for triggering object detection and tracking initialised.
    debugMode=False
    trackingEnabled=False

    #obtaining the object for using the video
    capture = cv2.VideoCapture('bouncingBall.avi')

    #if object not obtained then we cannot use the video!
    if not capture.isOpened():
        print("could not open video")
        return -1

    #read the first frame
    _,frame=capture.read()

    # We start our object detection and tracking from now on .
    while(int(capture.get(cv2.CAP_PROP_POS_FRAMES)) < int(capture.get(cv2.CAP_PROP_FRAME_COUNT))):# condition for checking whether video has ended or not.The next frame number should always be less than total numbe of frames in the video.

            #reading consecutive frames and turning them into gray-scale
            ok1,frame1=capture.read()
            grayImage1=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
            ok2, frame2=capture.read()
            grayImage2=cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            #taking the abolute difference between two frames
            differenceImage=cv2.absdiff(grayImage1,grayImage2)

            #thresholding the resultant difference of frames with the treshold value=20
            ret,thresholdImage=cv2.threshold(differenceImage,SENSITIVITY_VALUE,255,cv2.THRESH_BINARY)

            #using averaging filter of size 10 x 10 to smoothen the thresholded frames
            blurredImage=cv2.blur(thresholdImage,(BLUR_SIZE,BLUR_SIZE))

            #thresholding the blurred frame
            ret,thresholdImage=cv2.threshold(blurredImage,SENSITIVITY_VALUE,255,cv2.THRESH_BINARY)

            if debugMode==True:
                #uncomment the belowe line to see the thresholded image
                #cv2.imshow("Final Threshold Image",thresholdImage)
                trackingEnabled=True
            else:
                cv2.destroyWindow("Final Threshold Image")

            if trackingEnabled==True:
                searchForMovement(thresholdImage,frame1) #function to trace the object in the former frame read.

            cv2.imshow("Frame1",frame1) #the output of this file.

            debugMode=True #enabling debugMode which in turn will enable tracking mode

            # Using the below code we are able to control the display of result according to the
            #viewer preference . Press 'p' to pause and esc key to exit.
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

    #releasing the video-object and destroying all the windows.
    capture.release()
    cv2.destroyAllWindows()

#calling the main function.program starts from here.
if __name__=="__main__":
    mains()
















































