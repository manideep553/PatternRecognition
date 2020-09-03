# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 21:18:07 2019

@author: Rajesh
"""

import matplotlib.pyplot as plt 
import numpy as np
import math
from scipy import signal
import cv2
import pandas as pd
import matplotlib.image as mpimg 


# Function to apply gabor transformation.
def apply_wavelet_transform(sigma, theta, freq):
    
    filter_size = 10
    filter_half_size = int(filter_size / 2)
    [x, y] = np.meshgrid(range(-filter_half_size , filter_half_size+1), range(-filter_half_size , filter_half_size+1))
    
    pB = (-1) * (np.square(x) + np.square(y))
    pB = pB / (2 * sigma ** 2)
    g_sigma = (1 / (2 * np.pi * sigma ** 2)) * np.exp(pB)
    
    xcos = x * math.cos(theta)
    ysin = y * math.sin(theta)
    
    real_g = g_sigma * np.cos((2 * np.pi * freq) * (xcos + ysin))
    img_g = g_sigma * np.sin((2 * np.pi * freq) * (xcos + ysin))
    
    return real_g, img_g


# =============================================================================
# def extend_borders(img1, img2):
#     [row, col] = img2.shape
#     avg = np.average(img1)
#     for i in range(row):
#         for j in range(col):
#             if img2[i, j] == 0:
#                 img1[i, j] = avg
#     return img1            
# =============================================================================
    
# Function to extract the feature vector from an image.
def extract_features(input_image_name, output_image_name, mask_image_name):
    
    # Read an Image
    img = cv2.imread(input_image_name) 
    
    # Read output image
    output_img = mpimg.imread('E:\\Pattern_Recognition\\training\\training\\1st_manual\\' + output_image_name)
    
    msk = mpimg.imread('E:\\Pattern_Recognition\\training\\training\\mask\\' + mask_image_name)
    
    # split channels in an image.
    [b,g,r] = cv2.split(img)
    
    #g = extend_borders(g, msk)
    # convert green channel as features.
    features = g.flatten()
    features.shape = (len(features),1)
    features = (features - np.mean(features)) / np.std(features)
    
    # Invert green image
    inverted_green_Image = np.invert(g)
    plt.imshow(inverted_green_Image, cmap = 'gray')
    
    # Convert output values to binary values.
    output_label = output_img.flatten()
    output_label[output_label > 0] = 1
    
    # Scaling
    sigmas = [1, 2, 3, 4]
    
    # Angle
    thetas = [0.174533,	0.349066,	0.523599,	0.698132,	0.872665,	1.0472,	1.22173, 1.39626, 1.5708, 1.74533,
          1.91986, 2.0944, 2.26893, 2.44346, 2.61799, 2.79253, 2.96706]
    
    # Frequency
    freqs = [0.1, 0.2, 0.3] 
    
    final_vector = []
    
    for sigma in sigmas:
        df = pd.DataFrame()
        for theta in thetas:
            for freq in freqs:
                
                wavelet_output_real, wavelet_output_imag = apply_wavelet_transform(sigma, theta, freq)
                x = 'sigma = ' + str(sigma) + ' ' + 'theta = ' + str(theta) + ' ' + 'freq = ' + str(freq)
                plt.title(x)
                plt.imshow(wavelet_output_real, cmap = 'gray')
                
                A = signal.convolve2d(inverted_green_Image, wavelet_output_real, mode = 'same')
                B = signal.convolve2d(inverted_green_Image, wavelet_output_imag, mode = 'same')
                
                output_val = np.sqrt((A ** 2) + (B ** 2))
                col_name = str(theta) + ' ' + str(freq)
                df[col_name] = output_val.flatten()
                
            
        x = df.max(axis = 1)
        y = df.min(axis = 1)
        ans_vec = np.zeros(x.shape) 
        
        for i in range(len(x)):
            ans = x[i] if np.abs(x[i]) > np.abs(y[i]) else y[i]
            ans_vec[i] = ans
        final_vector.append(ans_vec)
            
    # Extract the feature vector.
    for nparray in final_vector:
        nparray.shape = (len(nparray), 1)
        nparray = (nparray - np.mean(nparray)) / np.std(nparray)
        features = np.concatenate((features, nparray), axis = 1)  
    return features, output_label    


# Input images 
input_image_list = ['21_training.tif', '22_training.tif', '23_training.tif', '24_training.tif', '25_training.tif',
                    '26_training.tif']

# Output images
output_image_list = ['21_manual1.gif', '22_manual1.gif', '23_manual1.gif','24_manual1.gif', '25_manual1.gif',
                     '26_manual1.gif']  

mask_list = ['21_training_mask.gif', '22_training_mask.gif', '23_training_mask.gif', '24_training_mask.gif', '25_training_mask.gif',
             '26_training_mask.gif'] 

final_feature_vector = []
label_vector = []

# For each image extract the feature vector and append to list. 
for input_image_name, output_image_name, mask_image_name in zip(input_image_list, output_image_list, mask_list):
    result, labels_list = extract_features(input_image_name, output_image_name, mask_image_name)
    final_feature_vector.append(result)
    label_vector.append(labels_list)
    
 
final_feature_vector = np.concatenate(final_feature_vector, axis = 0) 
label_vector = np.concatenate(label_vector, axis = 0)

# Test feature vector
test_feature_vector, test_labels = extract_features('27_training.tif', '27_manual1.gif', '27_training_mask.gif')  

# Naive bayes classifier.
from sklearn.naive_bayes import GaussianNB 
model = GaussianNB() 
model.fit(final_feature_vector, label_vector)
predictions = model.predict(test_feature_vector)
predictions[predictions == 1] = 255

out1 = mpimg.imread('E:\\Pattern_Recognition\\training\\training\\1st_manual\\27_manual1.gif')

predictions = predictions.reshape(out1.shape)
plt.imshow(predictions, cmap = 'gray')
   
  
# Random forest  
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=109) 
clf.fit(final_feature_vector, label_vector)

y_pred=clf.predict(test_feature_vector)  
y_pred = y_pred.reshape(out1.shape) 
plt.imshow(y_pred, cmap = 'binary')
    
    

    
          
    
    
    

    
    





 










 



    













            

   
            
        













