import cv2
import numpy as np
import easygui
#Open video
path=easygui.fileopenbox(filetypes=["*.mp4", "*.avi","*.mov"])
cap = cv2.VideoCapture(path)
#Detect background to substract
obj_detector = cv2.createBackgroundSubtractorMOG2()
# Properties to create new video
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
# Video Writer
video_writer = cv2.VideoWriter(path[:len(path)-4]+'_bk_erased.avi', cv2.VideoWriter_fourcc('M','J','P','G'),fps, (width, height))
#Detect background and erase on each frame
for frame_index in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    #Open frame on video
    ret,frame=cap.read()
    #Detect moving objects
    mask=obj_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 1, cv2.THRESH_BINARY)
    mask1=cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
    #Create new frame
    canvas=np.multiply(mask1,frame)
    cv2.imshow("",canvas)
    video_writer.write(canvas)
    # Breaking out of the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Close down everything
cap.release()
# Release video writer
video_writer.release()
cv2.destroyAllWindows()
