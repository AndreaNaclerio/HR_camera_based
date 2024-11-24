import torch
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from math import *

'''
distance : actual distance of the current pixel with respect to the central one
h : it's the scale, so the ROI dimension
'''
def gaussian_kernel(distance, h):
    distance_tensor = torch.tensor(distance, dtype=torch.float32)
    h_tensor = torch.tensor(h, dtype=torch.float32)
    weights = torch.exp(-0.5 * (distance_tensor / h_tensor) ** 2)
    total_weight = torch.sum(weights)

    normalized_weights = weights / total_weight
    return normalized_weights

'''
The couples are made such that we start from the farthest point.
'''
def neighborhood(x, y, angle, h):
    if angle == 0 or angle == 360:
        colum = [x] * (h+1)
        row = torch.arange(y-h, y+1).tolist()

    elif angle == 45:
        colum = torch.flip(torch.arange(x, x+h+1), dims=[0]).tolist()
        row = torch.arange(y-h, y+1).tolist()

    elif angle == 90:
        colum = torch.flip(torch.arange(x, x+h+1), dims=[0]).tolist()
        row = [y] * (h+1)

    elif angle == 135:
        colum = torch.flip(torch.arange(x, x+h+1), dims=[0]).tolist()
        row = torch.flip(torch.arange(y, y+h+1), dims=[0]).tolist()

    elif angle == 180:
        colum = [x] * (h+1)
        row = torch.flip(torch.arange(y, y+h+1), dims=[0]).tolist()

    elif angle == 225:
        colum = torch.arange(x-h, x+1).tolist()
        row = torch.flip(torch.arange(y, y+h+1), dims=[0]).tolist()

    elif angle == 270:
        colum = torch.arange(x-h, x+1).tolist()
        row = [y] * (h+1)

    elif angle == 315:
        colum = torch.arange(x-h, x+1).tolist()
        row = torch.arange(y-h, y+1).tolist()

    else:
        print(f"Angle is not valid")
        return

    return row, colum

'''
single iteration of LPA
'''
def LPA(im, x, y, row, column, h, channel):
    T = []
    S = []
    distances = []

    hh = max(abs(row[0]-y), abs(column[0]-x))
    for k in range(len(row)):
        i = column[k]
        j = row[k]
        T.append([1, hh, hh**2])
        S.append(im[j, i, channel])
        distances.append(hh)
        hh -= 1

    T = torch.tensor(T, dtype=torch.float32)
    S = torch.tensor(S, dtype=torch.float32)
    weights = gaussian_kernel(distances, h)
    W = torch.diag(weights)

    Q, R = torch.linalg.qr(T)
    try:
        R_inv = torch.linalg.inv(R)
    except torch.linalg.LinAlgError:
        R_inv = torch.linalg.pinv(R)

    X = R_inv @ Q.T @ W @ S
    return X, T, S, W, Q

'''
single iteration of the confidence interval
'''
def CI(S, W, Q, bw_im, h, channel, par):
    Q_tilde = torch.linalg.inv(W) @ Q
    g = torch.zeros(h+1, dtype=torch.float32)
    N = 2

    for i in range(N+1):
        g += Q_tilde[-1, i] * (W @ W @ Q_tilde[:, i])

    std_image = torch.std(torch.tensor(bw_im[:, :, channel], dtype=torch.float32))
    g_norm = torch.linalg.norm(g)
    std_f = std_image * g_norm

    value = torch.dot(g, S)
    low = value - par * std_f
    high = value + par * std_f

    return (low.item(), high.item())

'''
confidence interval decision
'''
def ICI(D, h_range):
    h_max = h_range[0]
    for i in range(1, len(h_range)):
        valid = True
        for j in range(0, i):
            if (D[h_range[i]][0] > D[h_range[j]][1]) or (D[h_range[i]][1] < D[h_range[j]][0]):
                valid = False
        if valid:
            h_max = h_range[i]
    return h_max

'''
apply LPA-ICA to the whole image
'''
def FULL(bw_im, y_coor, x_coor, par):
    FULL_FULL = dict()
    for y in range(y_coor, y_coor+1):
        for x in range(x_coor, x_coor+1):
            H_CHANNEL = []
            for channel in range(3):
                angles = [0, 45, 90, 135, 180, 225, 270, 315]
                ANGLE = dict()

                for angle in angles:
                    h_range = [2, 3, 5, 7, 11, 13, 15, 20]
                    D = dict()

                    for h in h_range:
                        row, column = neighborhood(x, y, angle, h)
                        X, T, S, W, Q = LPA(bw_im, x, y, row, column, h, channel)
                        D[h] = CI(S, W, Q, bw_im, h, channel, par)

                    ANGLE[angle] = ICI(D, h_range)

                H_CHANNEL.append(ANGLE)
            FULL_FULL[(y, x)] = H_CHANNEL
    return FULL_FULL

'''
determine boundary coordinates
'''
def coordinates(results):
    FINAL_COORD = dict()
    for coord in results.keys():
        single_res = results[coord]
        jj = coord[0]
        ii = coord[1]

        COORD = []
        for angle in single_res.keys():
            h = single_res[angle]
            if angle == 0:
                xx, yy = ii, jj-h
            elif angle == 45:
                xx, yy = ii+h, jj-h
            elif angle == 90:
                xx, yy = ii+h, jj
            elif angle == 135:
                xx, yy = ii+h, jj+h
            elif angle == 180:
                xx, yy = ii, jj+h
            elif angle == 225:
                xx, yy = ii-h, jj+h
            elif angle == 270:
                xx, yy = ii-h, jj
            elif angle == 315:
                xx, yy = ii-h, jj-h

            COORD.append((yy, xx))
        FINAL_COORD[coord] = COORD
    return FINAL_COORD

'''
merge scales (h) for different channels
'''
def merge_channels(final_res, y_coor, x_coor):
    results_merged = dict()
    for i in final_res[(y_coor, x_coor)][0].keys():
        temp = 0
        for j in range(3):
            temp += final_res[(y_coor, x_coor)][j][i]
        results_merged[i] = round(temp / 3)

    dict_results = dict()
    dict_results[(y_coor, x_coor)] = results_merged

    return dict_results






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
