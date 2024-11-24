import numpy as np
import cv2


############################################################
############################################################
##### IMPORT THE VIDEO USING THE FUNCTION load_video #######
############################################################
############################################################

'''
input: video and length cutting(in sec)
import video + cutting the portion
NB. the video is imported in RGB using the function BRG_RGB
'''
def load_video(video_link, length):
  [video, video_fps] = import_video(video_link)
  video = BGR_RGB(video)
  # compute the duration in sec of the video (information used in the GT_HR function)
  len = round(video.shape[0]*(1/video_fps))

  # consider only the first 20 sec of the video
  video_short = cut_video_length(video, video_fps, length, 'video_short')
  print('video cutted - ok')

  return video_short, len


'''
from BGR --> RGB 
'''
def BGR_RGB(video):
  video_RGB = []
  for i in video:
    video_RGB.append(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
  video_RGB = np.array(video_RGB)

  return video_RGB


'''
cut the video to make it shorter
video -> np.array
'''
def cut_video_length(video, fps, time, name_file):
    num_frame = round(fps)*time
    cutted_video = video[:num_frame] #it's the cutted video but stil in np.array format
    #array_to_video(cutted_video, name_file, fps)
    
    return cutted_video


'''
import video as np.array from .mp4 or .avi
input : video_name -> string + extension
output : image -> np.array
'''
def import_video(video_name):
    cap = cv2.VideoCapture(video_name) 
    #cap = cv2.VideoCapture('vid_s1_T1.avi')
    if (cap.isOpened()== False): 
        print("Error opening video file") 

    video_fps = cap.get(cv2.CAP_PROP_FPS) #get video FPS   
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #get number of frame

    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    video = [0]*frame_count
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            #print(frame.shape)
            video[i] = frame
            i += 1
        else:
            break
    cap.release() 

    return np.array(video), video_fps







