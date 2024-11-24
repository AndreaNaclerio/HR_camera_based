
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mediapipe
import statistics
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon
from math import *
from scipy.linalg import lstsq
#from SIGNAL_extraction_function import *
from VIDEO_load_function import *
from VIDEO_PreProcessing_function import *
from UBFC_RPPG_function import *
from google.colab.patches import cv2_imshow


'''
distance : actual distance of the current pixel with respect to the central one
h : it's the scale, so the ROI dimension
NB. for the weight computation, we have the factor ditance/h to compare the distance to the scale, and according to the ratio, assign the correct weight
'''
def gaussian_kernel(distance, h):
    weights =  np.exp(-0.5 * (distance / h) ** 2) #it returns the weight associated to the current neighbor pixel
    total_weight = np.sum(np.array(weights))

    # Step 3: Normalize the weights
    normalized_weights = weights / total_weight
    return normalized_weights

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

'''
The couples are made such that we start from the farest point.
For this reason, in the weigth assigment, we start from the lowest one.
'''
def neighborhood(x, y, angle, h): #define specific angle
  if angle == 0 or angle == 360:
    colum = [x]*(h+1)
    row = np.arange(y-h,y+1)

  elif angle == 45:
    side = h
    colum = np.flip(np.arange(x,x+side+1)) # arange create list of values only if they are growing. So to invert it, we use the function flip
    row = np.arange(y-side,y+1)

  elif angle == 90:
    colum = np.flip(np.arange(x,x+h+1))
    row = [y]*(h+1)

  elif angle == 135:
    side = h
    colum = np.flip(np.arange(x,x+side+1))
    row = np.flip(np.arange(y,y+side+1))

  elif angle == 180:
    colum = [x]*(h+1)
    row = np.flip(np.arange(y,y+h+1))

  elif angle == 225:
    side = h
    colum = np.arange(x-side,x+1)
    row = np.flip(np.arange(y,y+side+1))

  elif angle == 270:
    colum = np.arange(x-h,x+1)
    row = [y]*(h+1)

  elif angle == 315:
    side = h
    colum = np.arange(x-side,x+1)
    row = np.arange(y-side,y+1)

  else:
    print(f"Angle is not valid")
    return

  return row, colum

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

'''
single iteration of LPA -> fixed pixel(x,y) and h (determied by row and column given by the neighbrhood function)
'''
def LPA(im,x,y,row,column,h,channel):
  T = [] # Tx return the values calculated through the polynomial
  S = [] # vector true values
  distances = []

  hh = max(abs(row[0]-y),abs(column[0]-x))
  for k in range(len(row)):
    i = column[k]
    j = row[k]
    T.append([1, hh, hh**2]) # though we are working with images, we are considering a fitting along a line (with a particular direction)
    S.append(im[j, i, channel]) # <---- SELECT A SPECIFIC CHANNEL, SO WE ARE IN THE SAME SCENARIO AS THE GRAY-SCALE / BLACK-WHITE
    distances.append(hh)
    hh -= 1

  #building the working matrix
  T = np.array(T) #T:matrix with structure of polynomial
  S = np.array(S)
  weights = gaussian_kernel(np.array(distances), h) # i built the coordinates such that the first ones are the farest, so higher distances are the first ones so the correspondent weights are low
  W = np.diag(weights)

  #compute the best parameters solving the least square problem
  Q,R = np.linalg.qr(T) # Q-R decomp

  # in some cases the matrix inverse does not exist
  try:
    R_inv = np.linalg.inv(R)
  except np.linalg.LinAlgError:
    R_inv = np.linalg.pinv(R)

  X =  R_inv @ Q.T @ W @ S #X:coefficient  S:true values  .T -> transpose

  return X, T, S, W, Q

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

'''
single iteration of the confidence interval -> fixed pixel(x,y) and h (determied by row and column given by the neighbrhood function)
'''
def CI(S,W,Q,bw_im,h,channel, par):
  Q_tilde = np.linalg.inv(W) @ Q

  g = np.zeros(h+1)
  N = 2 #polynomial degree

  #print(Q_tilde.shape,W.shape)


  for i in range(N+1):
    #print(Q_tilde[-1,i],(W**2).shape, (Q_tilde[:,i]).shape)
    g += Q_tilde[-1,i] * (W**2 @ Q_tilde[:,i])


  std_image = np.std(bw_im[:,:,channel])
  g_norm = np.linalg.norm(g)
  std_f = std_image*g_norm

  par =  par  #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  value = np.dot(g,S)
  low = value - par*std_f
  high = value + par*std_f

  return (low, high)

'''
chatgpt initial solution (not the correct one)
'''
def CI_chat(S,X,T):
  values = T @ X
  residuals = S - values
  variance = np.var(residuals)

  return variance

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

def ICI(D,h_range): # D: contains the confidence interval for the different h
  h_max = h_range[0]
  for i in range(1,len(h_range)): # we are studying the current one (an higher h)
    Bool = True
    for j in range(0,i): # we are retrivinf the previous condifene interval related to all the lower h (with respect the current one)
      #if not (((D[h_range[i]][0] > D[h_range[j]][0]) and (D[h_range[i]][0] < D[h_range[j]][1])) or ((D[h_range[i]][1] > D[h_range[j]][0]) and (D[h_range[i]][1] < D[h_range[j]][1]))):
      #  Bool = False
      if (D[h_range[i]][0] > D[h_range[j]][1]) or (D[h_range[i]][1] < D[h_range[j]][0]):
        Bool = False

    if Bool:
      #<print('in')
      h_max = h_range[i]

  return h_max

'''
chatgpt initial solution (not the correct one)
'''
def ICI_chat(D,h_range): # D: contains the confidence interval for the different h
  best_estimate = None
  best_confidence_interval = float('inf')
  for h in h_range: # we are studying the current one (an higher h)
    if (D[h] <= best_confidence_interval):
      best_confidence_interval = D[h]
      best_estimate = h

  return best_estimate

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

'''
Extention to the whole image. For each pixel we can compute the LPA-ICA.
For the moment, the for loop is limited to only one point since there is no need to compute it for all the pixels
'''

def FULL(bw_im,y_coor,x_coor,par):
  #for jj in range(bw_im.shape[1]):
  #  for ii in range(bw_im.shape[0])

  # variable
  FULL_FULL = dict() #store h associated to all the pixel

  # selecting the pixel
  for y in range(y_coor,y_coor+1): #260
    for x in range(x_coor,x_coor+1): #330
      H_CHANNEL = []
      for channel in range(3):
        angles = [0,45,90,135,180,225,270,315]
        #angles = [0]
        ANGLE = dict() # store of the h associated to the different angle (fixed pixel)

        # selecting the angle (fixed the pixels)
        for angle in angles:
          h_range = [2,3,5,7,11,13,15,20] # era fino a 15,20
          D = dict() # CI store (fixed pixel and anglem different h)

          # selecting the scale (fixed pixel and angle)
          for h in h_range:
            row, column = neighborhood(x, y, angle, h)
            X, T, S, W, Q = LPA(bw_im,x,y,row,column,h,channel)
            #print(S)

            D[h] = CI(S,W,Q,bw_im,h,channel,par)
            #D[h] = CI_chat(S,X,T)

          #print(D)

          ANGLE[angle] = ICI(D, h_range)
          #ANGLE[angle] = ICI_chat(D, h_range)


        H_CHANNEL.append(ANGLE)
      FULL_FULL[(y,x)] = H_CHANNEL

  return FULL_FULL


#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
'''
given the central pixel and the h(scale) along the different directions, we compute the boundary coordinates
'''

def coordinates(results):
  FINAL_COORD = dict()
  for coord in results.keys():
    single_res = dict(results)
    single_res[coord] = single_res[coord] #<-- IF WE MERGE THE 3 RGB, WE DISCARD THE SELECTION OF THE CHANNEL
    #single_res[coord] = single_res[coord][1] # ATTENTION define channel we want to analyse
    jj = coord[0]
    ii = coord[1]

    COORD = []
    for angle in single_res[coord].keys():
      h = single_res[coord][angle]
      #print(angle, h)

      if angle == 0:
        xx = ii
        yy = jj-h
      elif angle == 45:
        xx = ii+h
        yy = jj-h
      elif angle == 90:
        xx = ii+h
        yy = jj
      elif angle == 135:
        xx = ii+h
        yy = jj+h
      elif angle == 180:
        xx = ii
        yy = jj+h
      elif angle == 225:
        xx = ii-h
        yy = jj+h
      elif angle == 270:
        xx = ii-h
        yy = jj
      elif angle == 315:
        xx = ii-h
        yy = jj-h

      COORD.append((yy,xx))
    FINAL_COORD[coord] = COORD

  return FINAL_COORD


#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
'''
plot the polygon connecting the points along the differnt directions
'''
# input the coordinates (so coordinates function output)
def plot_2(c,bw_im):
  #x = []
  #y = []
  for k,l in c.keys(): # here we can indicated the point that we want to examinate (dictionary keys)
    count = 0
    x = []
    y = []
    for f,d in c[(k,l)][1:]:
      y.append((c[(k,l)][count][0], f))
      x.append((c[(k,l)][count][1], d))
      count += 1

    y.append((c[(k,l)][-1][0], c[(k,l)][0][0]))
    x.append((c[(k,l)][-1][1], c[(k,l)][0][1]))

    print(x)
    print(y)

    # ATTENTION : QUANDO PLOT, L'ORDINE è X,Y
    fig, ax = plt.subplots()
    ax.imshow(bw_im)
    for temp in range(len(x)):
      ax.plot(x[temp],y[temp], label='Line 1', color='r')
    ax.scatter(l, k, color='g')

    #ax.set_xlim(300, 500)
    #ax.set_ylim(250, 450)  # Invert y-axis for correct orientation in images

    #ax.invert_yaxis()

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
'''
plot the star like plot, non connecting the point
'''

# input the coordinates (so coordinates function output)
def plot(c, bw_im):
  x = []
  y = []
  for k,l in c.keys(): # here we can indicated the point that we want to examinate (dictionary keys)
    y.append([k,k])
    x.append([l,l])
    for f,d in c[(k,l)]:
     y.append([k,f])
     x.append([l,d])

  # ATTENTION : QUANDO PLOT, L'ORDINE è X,Y
  fig, ax = plt.subplots()
  ax.imshow(bw_im)
  for temp in range(len(x)):
    ax.plot(x[temp],y[temp], label='Line 1', color='r')
  ax.scatter(x[0][0], y[0][0], color='g')

  #ax.set_xlim(300, 500)
  #ax.set_ylim(250, 450)  # Invert y-axis for correct orientation in images

  #ax.invert_yaxis()

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

'''
results: it's the vector contained the boundary pixel coordinates
return the pixel coordinates within the region that has been identified
'''
def pixel(results,bw_im):
  POINTS = set() # contain all the pixels merging the different areas

  for i in results.keys(): # for each single pixel within the dictionary
    polygon = Polygon(results[i]) # bulding the polygon defined by the points

    image_height = bw_im.shape[0]
    image_width = bw_im.shape[1]


    x, y = np.meshgrid(np.arange(image_width), np.arange(image_height))
    grid_points = np.vstack([x.ravel(), y.ravel()]).T

    inside_mask = np.array([polygon.contains(Point(p)) for p in grid_points])
    inside_points = grid_points[inside_mask]

    POINTS = POINTS.union(set(map(tuple, inside_points)))
  return POINTS

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

'''
merge the scale(h) found for the different channels (simple mean)
'''
def merge_channels(final_res,y_coor,x_coor):
  results_merged = dict()
  for i in final_res[(y_coor,x_coor)][0].keys(): #all the angles
    temp = 0
    for j in range(3): #3 channels
      temp += final_res[(y_coor,x_coor)][j][i]
    results_merged[i] = round(temp/3)

  dict_results = dict()
  dict_results[(y_coor,x_coor)] = results_merged

  return dict_results

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

'''
extract the identified region, cropping a small portion.
The pixel selection is done on the first frame, and then it's extent to all the video (multiple frame, same initial pixel)
(otherwise, a more complete method shold be to apply it for each frame, and at each iteration we should select a starting pixel)
'''
def extract_region(pix_s,video):
  pix = np.array(list(pix_s))
  min_x = np.min(pix[:, 1])
  max_x = np.max(pix[:, 1])
  min_y = np.min(pix[:, 0])
  max_y = np.max(pix[:, 0])

  num_frame = video.shape[0]
  null_pixel = 0
  ROI = np.zeros((num_frame,max_y-min_y,max_x-min_x,3),int)
  for i in np.arange(min_x,max_x):
    for j in np.arange(min_y,max_y):
      curr = (j,i)
      if (curr in pix_s):
        for k in range(num_frame): #fixed pixel (extracting the information moving between the differnt frames)
          ROI[k,j-min_y,i-min_x] = video[k,j,i] #we are working directly on the video (multiple frames)
      else:
        null_pixel += 1
        for k in range(num_frame):
          ROI[k,j-min_y,i-min_x] = [0,0,0]
  
  return ROI, null_pixel
