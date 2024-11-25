# Roi's exctraction adopting the mediapipe framework

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mediapipe
import statistics

#from SIGNAL_extraction_function import *
from VIDEO_load_function import *
from VIDEO_PreProcessing_function import *
#from UBFC_RPPG_function import *
from google.colab.patches import cv2_imshow


'''
ROI segmentation on a single image, specifing which region we are interesting in (roi parameter)
The roi's countours have been selected manually, connecting several landmarks
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

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################


'''
Count the pixels whose values is != from 0. This sis needed since the output image of the skim_segmentation function has the same dimension of the starting one. 
In this way it's possible to crop only the region of our interest.

conta e trova i pixel che sono diversi da zeri partendo dall'immagine processata dalla skin segmentation per poter definire la posizione e la dimensione del crop per tagliare l'immagine
altrimenti l'immagine risultante dal skin segmentation ha la stessa dimensione e tutta la parte non segmentata è nera
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


#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

'''
skin_segmentation extention to the entire video (several frames)
roi: insert the roi's name (fore_head, cheek_dx, cheek_sx, chin)
tensor_shape: 
estensione del skin segmentation non ad una sola immagine, ma viene applicata ad un video -> quindi abbiamo richiamato la funzione definita sopra per astenderla a tutti i frame del video
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
