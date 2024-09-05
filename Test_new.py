import os

import cv2
import numpy as np
import pandas as pd

# Load the full image
full_image = cv2.imread(r'C:\Users\SESA737873\Downloads\BFR\BFR\train\a.JPG', 0)

# Initialize result list
results = []
parts_folder=r'C:\Users\SESA737873\Downloads\BFR\BFR\test'
# Iterate over the images in the parts folder
for filename in os.listdir(parts_folder):
    # Load the part image
    part_image = cv2.imread(os.path.join(parts_folder, filename), 0)
    
    # Perform template matching
    if part_image.shape[0] > full_image.shape[0] or part_image.shape[1] > full_image.shape[1]:
                part_image = cv2.resize(part_image, (full_image.shape[1], full_image.shape[0]))
    res = cv2.matchTemplate(full_image, part_image, cv2.TM_CCOEFF_NORMED)
    
    # Set a threshold
    threshold = 0.4
    
    # Find where the part image matches the full image
    loc = np.where(res >= threshold)
    
    # If a match is found
    if len(loc[0]) > 0:
        # Append the result to the results list
        results.append([filename, 1])
        
        # Draw a rectangle around the matched region
        # for pt in zip(*loc[::-1]):
        #     cv2.rectangle(full_image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            
        # Display the result
        # cv2.imshow('Detected', full_image)
        # cv2.waitKey(1)
    else:
        # Append the result to the results list
        results.append([filename, 0])

# Convert the results to a DataFrame
df = pd.DataFrame(results, columns=['Image', 'Match'])

# Write the DataFrame to a CSV file
df.to_csv('results.csv', index=False)
