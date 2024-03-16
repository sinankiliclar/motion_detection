import cv2

cap=cv2.VideoCapture("D:\opencv_udemy/18_motion_detection/runner.mp4")

ret,frame1=cap.read()
ret,frame2=cap.read()
color=(0,255,0)
while 1:
    
    diff=cv2.absdiff(frame1,frame2)
    cv2.imshow("Diff",diff)

    gray=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    cv2.imshow("Blur",blur)

    _,tresh=cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated=cv2.dilate(tresh,None,3)   
    cv2.imshow("Dilated",dilated)
    
    contours,_=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame1,contours,-1,color,3)    
    
    cv2.imshow("Video",frame1)
    frame1=frame2
    ret,frame2=cap.read()


    if cv2.waitKey(30) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()