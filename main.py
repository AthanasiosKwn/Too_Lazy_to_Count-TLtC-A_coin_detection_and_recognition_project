import cv2 as cv
import matplotlib.pyplot as plt
import json
import os
import numpy as np


# Create coin database:

# A. Radiuses as a simple but limited descriptor

# Folder path containing ground truth images
folder_path = r'C:\Users\xthan\Desktop\portofolio_projects\coins_recognition\ground_truth_coins'


def create_image_dict(folder_path):
    '''Input the folder path and receive a dictionary of images'''
    # Dictionary containing the grayscale ground truth images
    imgs = {}


    # Iterate through the folder components - images
    for image_name in os.listdir(folder_path):

        # Join folder path with image name
        image_path = os.path.join(folder_path, image_name)

        # Load the image and append to dictionary
        if image_name.endswith('.jpg') or image_name.endswith('.png') or image_name.endswith('.jpeg'):
            image = cv.imread(image_path)
            

            if image is None:
                print(f"Failed to load image '{image_path}'.")
                continue

        
            imgs[image_name.split('.')[0]] = image
    return imgs
imgs = create_image_dict(folder_path)


# Segment the image, detect each coin and compute the radius descriptor of it by fitting a minimun bounding circle

def radius_extractor(imgs):
    # Initialize the dictionary that will contain the radius for each different coin. The radius will act as a descriptor
    radiuses = {}

    # Iterate through the images
    for key, image in imgs.items():

        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur_img = cv.GaussianBlur(image_gray, (21, 21), 0)
        
        # Threhold image through adaptive binary thresholding
        thresholded = cv.adaptiveThreshold(blur_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV, 11, 1) # binary inv because there is a white background
        
        # Construct structuring element used for morphological operations
        kernel = np.ones((7, 7), np.uint8)

        # Apply morphological opening to remove small white noise pixels
        opened_image = cv.morphologyEx(thresholded, cv.MORPH_OPEN, kernel, iterations=1)

        # Apply morphological closing to fill in the coin area
        closed_image  = cv.morphologyEx(opened_image, cv.MORPH_CLOSE, kernel, iterations=20)
        
        # Find external contours in the binary image
        contours, hierarchy = cv.findContours(closed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
       
        # Sort contours by descending contour area
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        if len(contours) == 0:
            print("No contours found in the image")
            continue
        
        # Fit the minimum enclosing circle on the contour with the biggest area and obtain the radius - descriptor of the coin
        (x, y), radius = cv.minEnclosingCircle(contours[0])
        cv.circle(image, (int(x),int(y)), int(radius), (0,255,0), 5)
        plt.imshow(cv.cvtColor(image,cv.COLOR_BGR2RGB))
        plt.show()

        radiuses[key] = radius
    return radiuses

radiuses = radius_extractor(imgs)
exit()


# Change dictionary keys of 'radiuses' dictionary into the corresponding coin values

# Define the mapping of old keys to new keys. 
key_mapping = {'euro001': 0.01, 'euro002': 0.02, 'euro005': 0.05, 'euro010':0.10, 'euro020':0.20, 'euro050':0.50, 'euro1':1.0, 'euro2':
               2.0}

# Create a new dictionary with updated key names
radiuses_keys_updated = {key_mapping.get(k, k): v for k, v in radiuses.items()}
   

# Save dictionary to a JSON file
with open('radiuses.json', 'w') as json_file:
    json.dump(radiuses_keys_updated, json_file, indent=4)


# B. SIFT 

def sift_extractor(imgs):
    # The following dictionary will contain eight key-value pairs, one for each type of coin
    # The key will be the value of the coin, the value will be the SIFT descriptors of the corresponding coin image
    sift_database = {}

    # Same as the above dict but the values are key points detected and not their descriptors
    key_points = {}

    # Iterate through the images as before, segment the coin in each image and calculate the SIFT desriptors only at the coin region
    for key, image in imgs.items():

        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur_img = cv.GaussianBlur(image_gray, (21, 21), 0)
        
        # Threshold image through adaptive binary thresholding
        thresholded = cv.adaptiveThreshold(blur_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV, 11, 1) # binary inv because there is a white background
        
        # Construct structuring element used for morphological operations
        kernel = np.ones((7, 7), np.uint8)

        # Apply morphological opening to remove small white noise pixels
        opened_image = cv.morphologyEx(thresholded, cv.MORPH_OPEN, kernel, iterations=1)

        # Apply morphological closing to fill in the coin area
        closed_image  = cv.morphologyEx(opened_image, cv.MORPH_CLOSE, kernel, iterations=20)

        # Find external contours in the binary image
        contours, hierarchy = cv.findContours(closed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # Sort contours by descending contour area
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        if len(contours) == 0:
            print("No contours found in the image")
            continue
        
        # Fit the minimum enclosing circle on the contour with the biggest area and obtain the radius - descriptor of the coin
        (x, y), radius = cv.minEnclosingCircle(contours[0])

        # Create mask. It will be used to specify to what part of the image the SIFT algorithm is applied aka only the coin region 
        circle_mask = np.zeros(image_gray.shape,dtype=np.uint8)

        # Draw a white circle (filled) on the mask based on the center and the radius of the coin detected
        cv.circle(circle_mask, (int(x),int(y)), int(radius),(255), -1)
    
        # Create SIFT detector
        sift = cv.SIFT_create()

        # Detect feature points and compute the corresponding SIFT descriptors
        keypoints, sift_descriptors = sift.detectAndCompute(image_gray, circle_mask)

        sift_database[key] = sift_descriptors
        key_points[key] = keypoints
    return sift_database, key_points

sift_database, key_points = sift_extractor(imgs)


# Create a new dictionary with updated key names
sift_database_updated = {key_mapping.get(k, k): v for k, v in sift_database.items()}
key_points_updated = {key_mapping.get(k, k): v for k, v in key_points.items()} 

# Convert ndarray descriptors to lists in order to save to JSON format
descriptors_converted = {label: des.tolist() for label, des in sift_database_updated.items()}

# Save dictionary to a JSON file
with open('sift_descriptors.json', 'w') as json_file:
    json.dump(descriptors_converted, json_file, indent=4)


# Convert KeyPoints to a serializable format. When SIFT is applied to an image a list of key points objects is returned.
# They key point object must be converted into a serializale format, a dictionary for example

def keypoints_to_list(keypoints):
    return [{'pt': kp.pt, 'size': kp.size, 'angle': kp.angle,
             'response': kp.response, 'octave': kp.octave, 'class_id': kp.class_id}
            for kp in keypoints]

keypoints_converted = {label: keypoints_to_list(points) for label, points in key_points_updated.items()}

# Save to JSON file
with open('keypoints.json', 'w') as json_file:
    json.dump(keypoints_converted, json_file, indent=4)


