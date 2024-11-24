#pip install mediapipe #<- install mediapipe 
import mediapipe
import scipy
import scipy.io
from scipy.signal import butter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import cv2

'''
WORKFLOW:
1. import the video
2. skin segmentation (select ROI) (skin_segmentation_video)
3. cropping the portion of the image valid
4. extract the signal from the video (one value for each frame) (get_signal)
5. plot signal + filtering
'''



###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
'''
SKIN SEGMENTATION
select the portion of the face we want to crop between: fore-head, cheeks sx and cheek dx
Input: single image (np.array)
roi: fore_head, cheek_sx or cheek_dx
Output: same dimension but with the region outside the roi is blacked
'''
def skin_segmentation(image, roi):
  mp_face_mesh = mediapipe.solutions.face_mesh
  face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
  results = face_mesh.process(image)
  landmarks = results.multi_face_landmarks[0]

  #define the roi to segment
  if roi == 'face':
    face_oval = mp_face_mesh.FACEMESH_FACE_OVAL

    df = pd.DataFrame(list(face_oval), columns = ["p1", "p2"])
    routes_idx = []
    p1 = df.iloc[0]["p1"]
    p2 = df.iloc[0]["p2"]

    for i in range(0, df.shape[0]):
      obj = df[df["p1"] == p2]
      p1 = obj["p1"].values[0]
      p2 = obj["p2"].values[0]
      route_idx = []
      route_idx.append(p1)
      route_idx.append(p2)
      routes_idx.append(route_idx)

  if roi == 'fore_head':
    fore_head = [(103,104),(104,105),(105,66),(66,107),(107,9),(9,336),(336,296),(296,334),(334,333),(333,332),(332,297),(297,338),(338,10),(10,109),(109,67),(67,103)]
    routes_idx = fore_head

  if roi == 'cheek_sx':
    cheek_sx = [(117,123),(123,187),(187,207),(207,216),(216,206),(206,203),(203,142),(142,100),(100,120),(120,119),(119,118),(118,117)]
    routes_idx = cheek_sx

  if roi == 'cheek_dx':
    cheek_dx = [(346,352),(352,411),(411,427),(427,436),(436,426),(426,423),(423,371),(371,329),(329,349),(349,348),(348,347),(347,346)]
    routes_idx = cheek_dx

  if roi == 'chin':
    chin = [(194,32),(32,140),(140,171),(171,175),(175,396),(396,369),(369,262),(262,418),(418,421),(421,200),(200,201),(201,194)]
    routes_idx = chin

  #############################################################################
  #extaction of the ROI
  routes = []
  for source_idx, target_idx in routes_idx:

    source = landmarks.landmark[source_idx]
    target = landmarks.landmark[target_idx]
    relative_source = (int(image.shape[1] * source.x), int(image.shape[0] * source.y))
    relative_target = (int(image.shape[1] * target.x), int(image.shape[0] * target.y))
    #cv2.line(image, relative_source, relative_target, (255, 255, 255), thickness = 2)
    routes.append(relative_source)
    routes.append(relative_target)

  mask = np.zeros((image.shape[0], image.shape[1]))
  mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
  mask = mask.astype(bool)
  out = np.zeros_like(image)
  out[mask] = image[mask]

  return out


###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

'''
conta e trova i pixel che sono diversi da zeri partendo dall'immagine processata dalla skin segmentation per poter definire la posizione e la dimensione del crop per tagliare l'immagine
altrimenti l'immagine risultante dal skin segmentation ha la stessa dimensione e tutta la parte non segmentata è nera
Input: image processed by the skin segmentation that has the same dimension but with only a small part where the pixels are valid 
'''
def valid_image(image):
    # Create a mask of non-zero pixels
    mask = np.any(image != 0, axis=-1)

    # Get the coordinates of non-zero pixels
    non_zero_coords = np.argwhere(mask)

    # Extract the minimum and maximum coordinates
    min_x, min_y = np.min(non_zero_coords, axis=0)
    max_x, max_y = np.max(non_zero_coords, axis=0)

    return min_x, max_x, min_y, max_y


###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

'''
estensione del skin segmentation non ad una sola immagine, ma viene applicata ad un video -> quindi abbiamo richiamato la funzione definita sopra per astenderla a tutti i frame del video
the skin segmentation is extended to all the frames of a video
number_valid_pixel -> we need to count the valid pixel so when we extract the signal from the video we can compute the average dividing by the actual number of valid pixel
'''
def skin_segm_video(video, roi, tensor_shape=True):
  cropped_image = []
  heigth = 0
  width = 0
  for iii in range(video.shape[0]):
    temp = skin_segmentation(video[iii], roi)
    [min_x, max_x, min_y, max_y] = valid_image(temp)

    temp_width = max_x - min_x
    temp_heigth = max_y - min_y

    if temp_heigth > heigth:
      heigth = temp_heigth
    if temp_width > width:
      width = temp_width

    cropped_image.append(temp)
  cropped_image = np.array(cropped_image)
  print('SEGMENTATION: DONE')

  final_image = []
  number_valid_pixel = []
  for k in range(cropped_image.shape[0]):
    [min_x, max_x, min_y, max_y] = valid_image(cropped_image[k])

    # to count the pixels != 0 so in the mean we consider only the number of valid pixels
    mask = np.any(cropped_image[k] != 0, axis=-1)
    non_zero_coords = np.argwhere(mask).shape[0]
    number_valid_pixel.append(non_zero_coords)

    if tensor_shape:
      final_image.append(cropped_image[k, min_x:min_x+width ,min_y:min_y+heigth , :])
    else:
      final_image.append(cropped_image[k, min_x:max_x, min_y:max_y , :])

  print('CROPPING: DONE')

  return np.array(final_image), number_valid_pixel

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

'''
given the cropped frames, from each one we want to extract the signal (one value from each frame)
output -> the different channels (RGB) of the same video are saved in the same dictionary
(this is for a specific ROI)
'''
# EXTRACT THE SIGNAL FROM THE CROPPED IMAGES (AVERAGE OF THE PIXEL VALUES) (FOR PLOTTING SIGNAL ARE SAVED IN )
# comodo salvare tutti i diversi segnali associato ad uno stesso video in un singolo dizionario
def signal_extraction_plotting(cropped_video, num_valid_pixel):
  signal = {'R':[], 'G':[], 'B':[]}
  for i in range(cropped_video.shape[0]):
    signal['R'].append(np.sum(cropped_video[i,:,:,0])/num_valid_pixel[i])
    signal['G'].append(np.sum(cropped_video[i,:,:,1])/num_valid_pixel[i])
    signal['B'].append(np.sum(cropped_video[i,:,:,2])/num_valid_pixel[i])
  
  return signal

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

'''
filtare il segnale con il passabasso utilizzando il range fisiologico del battito cardiaco
'''
def PB_filt(signal, LPF=0.75, HPF=2.5, fs=30):
  [b, a] = butter(1, [LPF / fs * 2, HPF / fs * 2], btype='bandpass')
  output = scipy.signal.filtfilt(b, a, np.double(signal))

  return output

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

'''
rappresentazione di diversi segnali in un grafico 3D per poterli comparare ma mantenendoli distinti
input: croppred videe and number of valid pixel
+ save dictionary signal in a CSV file

input can be:
- video -> so we can plot the RGB signal we can retrieve from the frames (Video= True)
- signal saved as dictionary (Video= False)
- args è il num_valid_pixel che nel caso di segnali non serve, quindi deve essere inserito come ultimo parametro in quanto è opzionale (prima di quelli già assegnati)
'''

def plot_3d(input, name, *args, Video=True, filtering=False):

  #extract the signal from the cropped video
  if Video:
    num_valid_pixel = np.array(args)[0]
    signal = signal_extraction_plotting(input, num_valid_pixel)
  else:
    signal = input
  
  #filtering if needed
  if filtering:
    signal['G'] = PB_filt(signal['G'])
    signal['R'] = PB_filt(signal['R'])
    signal['B'] = PB_filt(signal['B'])

  # Create the figure and axis
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  t = range(0,len(signal[list(signal.keys())[0]]))
  # Plot the data

  if Video:
    for i, color in enumerate(['R', 'G', 'B']):
      x = t
      z = signal[color]
      y = np.full_like(t, i)  # Create distinct z-values to separate the lines
      ax.plot(x, y, z, label=color, color=color.lower())
  else:
    for i, key in enumerate(signal.keys()):
      x = t
      z = signal[key]
      y = np.full_like(t, i)  # Create distinct z-values to separate the lines
      ax.plot(x, y, z, label=key)

  # Add labels and legend
  ax.set_xlabel('frame')
  ax.set_ylabel('')
  ax.set_zlabel('signal')
  ax.set_title(name)
  ax.legend()


  ax.view_init(elev=40, azim=30)
  fig.tight_layout()


  # Show the plot
  plt.show()

  import csv
  with open('/content/drive/MyDrive/tesi_UBFC-RPPG/pre_processing/data/'+name +'.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in signal.items():
       writer.writerow([key, value])

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

'''
estrarre il background, ROI che non era considerata prima
'''
def background(video):
  back_image = []
  for i in range(video.shape[0]):
    back_image.append(video[i,:200,:200,:])

  return np.array(back_image)

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

'''
è uguale al precedente ma è stato aggiunto come output la routes per permettere di rappresentare le ROI considerate
'''
def skin_segmentation_boundaries(image, roi):
  mp_face_mesh = mediapipe.solutions.face_mesh
  face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
  results = face_mesh.process(image)
  landmarks = results.multi_face_landmarks[0]

  #define the roi to segment
  if roi == 'face':
    face_oval = mp_face_mesh.FACEMESH_FACE_OVAL

    df = pd.DataFrame(list(face_oval), columns = ["p1", "p2"])
    routes_idx = []
    p1 = df.iloc[0]["p1"]
    p2 = df.iloc[0]["p2"]

    for i in range(0, df.shape[0]):
      obj = df[df["p1"] == p2]
      p1 = obj["p1"].values[0]
      p2 = obj["p2"].values[0]
      route_idx = []
      route_idx.append(p1)
      route_idx.append(p2)
      routes_idx.append(route_idx)

  if roi == 'fore_head':
    fore_head = [(103,104),(104,105),(105,66),(66,107),(107,9),(9,336),(336,296),(296,334),(334,333),(333,332),(332,297),(297,338),(338,10),(10,109),(109,67),(67,103)]
    routes_idx = fore_head

  if roi == 'cheek_sx':
    cheek_sx = [(117,123),(123,187),(187,207),(207,216),(216,206),(206,203),(203,142),(142,100),(100,120),(120,119),(119,118),(118,117)]
    routes_idx = cheek_sx

  if roi == 'cheek_dx':
    cheek_dx = [(346,352),(352,411),(411,427),(427,436),(436,426),(426,423),(423,371),(371,329),(329,349),(349,348),(348,347),(347,346)]
    routes_idx = cheek_dx
  #if roi == 'back_ground':
    #todo

  #############################################################################
  #extaction of the ROI
  routes = []
  for source_idx, target_idx in routes_idx:

    source = landmarks.landmark[source_idx]
    target = landmarks.landmark[target_idx]
    relative_source = (int(image.shape[1] * source.x), int(image.shape[0] * source.y))
    relative_target = (int(image.shape[1] * target.x), int(image.shape[0] * target.y))
    #cv2.line(image, relative_source, relative_target, (255, 255, 255), thickness = 2)
    routes.append(relative_source)
    routes.append(relative_target)

  mask = np.zeros((image.shape[0], image.shape[1]))
  mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
  mask = mask.astype(bool)
  out = np.zeros_like(image)
  out[mask] = image[mask]

  return out, routes

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

'''
rappresentazione grafica delle ROI che sono state considerate
'''
def boundary(image):
  for i in ['fore_head','cheek_dx','cheek_sx']:
    [image_cutted, route] = skin_segmentation_boundaries(image, i)
  
    for i in range(0, len(route)-1, 1):
      plt.imshow(image)
      plt.plot([route[i][0],route[i+1][0]], [route[i][1],route[i+1][1]], 'r*-', linewidth=1, markersize=1)
  
  # backgroud ROI considered
  plt.plot([0,200], [0,0], 'r*-', linewidth=1, markersize=1)
  plt.plot([200,200], [0,200], 'r*-', linewidth=1, markersize=1)
  plt.plot([200,0], [200,200], 'r*-', linewidth=1, markersize=1)
  plt.plot([0,0], [200,0], 'r*-', linewidth=1, markersize=1)
