import csv
import os

import cv2
import numpy as np

# Load the complete puzzle image
complete_puzzle = cv2.imread(r'C:\Users\SESA737873\Downloads\BFR\BFR\train\a.JPG', 0)
complete_puzzle = cv2.resize(complete_puzzle, (1000, 1000))

# Initialize ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB in the complete puzzle image
kp1, des1 = orb.detectAndCompute(complete_puzzle, None)

# Define ROI in the complete puzzle image
roi = complete_puzzle[0:complete_puzzle.shape[0], 0:complete_puzzle.shape[1]]
# Get list of all puzzle piece images in the folder
folder = r'C:\Users\SESA737873\Downloads\BFR\BFR\test'
images = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

# Initialize BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Open CSV file in write mode
with open('results_1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(["Image Name", "Number of Matches", "Matching Percentage", "ROI Location"])

    # Loop through all images
    for image_name in images:
        # Load puzzle piece image
        puzzle_piece = cv2.imread(os.path.join(folder, image_name), 0)

        # Find the keypoints and descriptors with ORB in the puzzle piece image
        kp2, des2 = orb.detectAndCompute(puzzle_piece, None)

        # Match descriptors
        matches = bf.match(des1, des2)

        # Sort matches in the order of their distance
        matches = sorted(matches, key = lambda x:x.distance)

        # Calculate matching percentage
        matching_percentage = len(matches) / len(des2) * 100

        # Homography
        if len(matches) > 10:
            src_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = puzzle_piece.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            complete_puzzle = cv2.polylines(complete_puzzle,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        # Write results to CSV
        writer.writerow([image_name, len(matches), matching_percentage, "ROI Location"])

print("Results written to results.csv")
