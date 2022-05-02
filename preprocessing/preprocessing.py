import os
import cv2 
import shutil
import random  
import imageio                  
import numpy as np 
import pandas as pd
import imgaug as ia 
import pyrealsense2 as rs    
import imgaug.augmenters as iaa      
import matplotlib.pyplot as plt

def extract_frames2(filepath, filename, frame_type, destination, bag):
  try:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(filepath + filename, repeat_playback = False)
    if frame_type == 'depth':
      config.enable_stream(rs.stream.depth)#, 640, 480, rs.format.z16, 6) 
    elif frame_type == 'color':
      config.enable_stream(rs.stream.color)
    
    profile = pipeline.start(config)

    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(5):
      pipeline.wait_for_frames()
    
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    i = 0
    while True:
      print("Saving frame: ", i)
      frames = pipeline.wait_for_frames()
      if frame_type == 'depth':
        colorizer = rs.colorizer()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        cv2.imwrite(destination + 'depth/' + '_0322' + bag + str(i).zfill(6) + '.png', depth_image)
      elif frame_type == 'color':
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite(destination + 'color/' + '_bag'+ bag + str(i).zfill(6) + '.png', color_image)
      i += 1
  finally:
    pass
  
def apply_image_augmentations(src_filepath, dest_filepath, joint_type):
  # Define the first set of transformations to apply to the images
  seq_1 = iaa.Sequential([iaa.Fliplr(0.5), # Horizontal flip
                          iaa.GammaContrast((0.5, 2.0), per_channel = True), # Gamma constrast
                          iaa.AddToHueAndSaturation((-60, 60)), # Add to hue and saturation
                          iaa.AdditiveGaussianNoise(scale = (30, 90)) # Gaussian blur
                          ],
                         random_order = True) 

  seq_2 = iaa.Sequential([iaa.Flipud(0.5), # Vertical flip
                          iaa.SigmoidContrast(gain = (3, 10), # Sigmoid contrast
                                              cutoff = (0.4, 0.6), 
                                              per_channel=True),
                          iaa.AdditiveGaussianNoise(scale = (10, 60)), # Gaussian blur
                          iaa.Add((-40, 40), per_channel=0.5) # Add values to pixels
                          ],
                         random_order = True) 
  i = 0
  while True:
    for img in os.listdir(src_filepath):
      read_image = imageio.imread(src_filepath + img)
      if (joint_type in img) and ('b' not in img):
        # Apply the transformations randomly
        r = random.choice([-1, 1])
        if r == -1:
          # Apply first set of transformations
          image_aug = seq_1(image = read_image)
          imageio.imwrite(dest_filepath + '_' + str(i) + '_' + joint_type + '.png', 
                      image_aug)
        elif r == 1:
          # Apply second set of transformations
          image_aug = seq_2(image = read_image)
          imageio.imwrite(dest_filepath + '_' + str(i) + '_' + joint_type + '.png', 
                      image_aug)
      i += 1
