#single object tracking using optical flow . Manually written lukas-kanade algorithm
#Not working!


import cv2
import sys
import numpy as np
from scipy import signal

def main():

    window_size=3

    while True:

        capture=cv2.VideoCapture('bouncingBall.avi')

        if (not capture.isOpened()):
            print("could not open video!")
            sys.exit()

        np.set_printoptions(threshold=np.inf)

        #start with 2nd frame till second last frame
        while(capture.get(cv2.CAP_PROP_POS_FRAMES))<(capture.get(cv2.CAP_PROP_FRAME_COUNT)-2):

            ret,fr=capture.read()
            grayImg=np.float64(cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY))

            ret1 , fr1 =capture.read()
            grayImg1=np.float64(cv2.cvtColor(fr1,cv2.COLOR_BGR2GRAY))

            ret2, fr2 = capture.read()
            grayImg2 =np.float64(cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY))

            kernel_x=np.array([[-1.,1.],[-1.,1.]])
            kernel_y=np.array([[-1.,-1.],[1.,1.]])
            kernel_t=np.array([[1.,1.],[1.,1.]])

            grayImg1 = grayImg1 / 255.
            grayImg2 = grayImg2 / 255.
            grayImg = grayImg / 255.

            fx=signal.convolve2d(grayImg1,kernel_x,mode='same',boundary='symm')
            fy=signal.convolve2d(grayImg1,kernel_y,mode='same',boundary='symm')
            ft=np.zeros((int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
            ft=signal.convolve2d(grayImg2,kernel_t,mode='same',boundary='symm')+signal.convolve2d(grayImg1,-kernel_t,mode='same',boundary='symm')

            for i in range(0,int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))):
                for j in range(0,int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))):

                    ft[i,j]=grayImg2[i,j]-(2*grayImg1[i,j])+grayImg[i,j]

            u = np.zeros((int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
            v = np.zeros((int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))

            w=window_size/2

            for i in range(w,int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))-1):
                for j in range(w,int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))-1):

                    fx_temp = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
                    fy_temp = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
                    ft_temp = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()

                    print(fx_temp)

                    A=np.array([[sum(np.square(fx_temp)),sum(np.multiply(fx_temp,fy_temp))],[sum(np.multiply(fx_temp,fy_temp)),sum(np.square(fy_temp))]])
                    b=np.array([[-sum(np.multiply(fx_temp,ft_temp))],[-sum(np.multiply(fy_temp,ft_temp))]])

                    mew=np.matmul(np.linalg.inv(A),b).flatten()
                    u[i,j]=mew[0]
                    v[i,j]=mew[1]


            print(u,v)

        sys.exit()


if __name__ == '__main__':
    main()