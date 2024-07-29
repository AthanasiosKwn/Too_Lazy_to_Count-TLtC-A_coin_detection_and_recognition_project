import cv2 as cv
import matplotlib.pyplot as plt
import json
import math
import numpy as np
import os
from sklearn.cluster import KMeans
from time import time

# Adaptive segmentation function
def adaptive_segmentation(image):
        
        base_height = 3072
        base_iterations = 20

        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Get the dimensions of the image
        image_height, _ = image_gray.shape

        # Calculate the scaling based on the assumption of uniform scaling and that the input image is a scaled version of the ground truth image
        # It does not work for differences in scale as a result of changes in camera - scene distance
        scale = image_height / base_height

        # Apply Gaussian blur
        blur_img = cv.GaussianBlur(image_gray, (21, 21), 0)
        
        # Threhold image through adaptive binary thresholding
        thresholded = cv.adaptiveThreshold(blur_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV, 11, 1) # binary inv because there is a white background
        
        # Construct structuring element used for morphological operations
        # Define base kernel size
        base_kernel_size = (7, 7)  # Base size that worked well for ground truth images

        # Adjust kernel size based on scale factor. 
        adjusted_kernel_size = tuple(int(dim * scale) for dim in base_kernel_size)
        adjusted_kernel_size = (max(adjusted_kernel_size[0], 1), max(adjusted_kernel_size[1], 1)) #ensure at least 1x1 kernel

        # Apply morphological opening to remove small white noise pixels
        opened_image = cv.morphologyEx(thresholded, cv.MORPH_OPEN, adjusted_kernel_size, iterations=1)

        # Apply morphological closing to fill in the coin area. Dynamic number of iterations based on image scale in relation to ground truth images scale. The formula needs @TODO work
        closed_image  = cv.morphologyEx(opened_image, cv.MORPH_CLOSE, adjusted_kernel_size, iterations= int(base_iterations*scale))
        return closed_image


# Image segmentation based on Kmeans clustering in the RGB space
def kmeans_segmentation(image):
        
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Flatten the image into a 2D array where each row is a pixel and each column is a color channel
        pixels = image_rgb.reshape(-1, 3)

        # Get the dimensions of the image
        image_height, image_width, channels = image_rgb.shape

        # Apply KMeans clustering to the pixels with 2 centroids. 
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=2).fit(pixels)
        # Labels
        labels = kmeans.labels_
        # Centroids
        centroids = kmeans.cluster_centers_

        # Determine which cluster centroid is closer to white ([255, 255, 255]). Values of pixels clustered to this centroid are set to 0
        white = np.array([255, 255, 255])
        distances = np.linalg.norm(centroids - white, axis=1)
        background_label = np.argmin(distances)

        # Create a binary image based on the clustering results
        binary_image = (labels != background_label).astype(np.uint8)

        # Reshape the binary image to the original image dimensions
        binary_image = binary_image.reshape(image_height, image_width)

        # Convert the binary image to 8-bit
        binary_image = (binary_image * 255).astype(np.uint8)
        return binary_image

# Function to read test images
def read_test_images(folder_path):
    # List containing test images
    images = []

    # Iterate through the folder components - images
    for image_name in os.listdir(folder_path):

        # Join folder path with image name
        image_path = os.path.join(folder_path, image_name)

        # Load the image and append to list
        if image_name.endswith('.jpg') or image_name.endswith('.png') or image_name.endswith('.jpeg'):
            image = cv.imread(image_path)

            if image is None:
                print(f"Failed to load image '{image_path}'.")
                continue

            images.append(image)
    return images



# Read JSON file containing the coin descriptors. Each key is a coin value and the corresponding dict value is the radius of the coin
with open('radiuses.json', 'r') as json_file:
    radiuses = json.load(json_file)


# Folder path containing test images
folder_path = r'C:\Users\xthan\Desktop\portofolio_projects\coins_recognition'
images = read_test_images(folder_path)

folder_path_scaled = r'C:\Users\xthan\Desktop\portofolio_projects\coins_recognition\scaled_test_images'
scaled_images = read_test_images(folder_path_scaled)

folder_path_adjusted = r'C:\Users\xthan\Desktop\portofolio_projects\coins_recognition\brightness_reduced_test_images'
brightness_adjusted_images = read_test_images(folder_path_adjusted)

folder_path_web = r'C:\Users\xthan\Desktop\portofolio_projects\coins_recognition\web_images'
web_images = read_test_images(folder_path_web)


# Coin detection and recognition function
def detect_recognize_radiuses(images, radiuses, kmeans=True):
    t1 = time()

    # Iterate through test images
    for image in images:
        
        # Base values of reference image (not scaled not brightness adjusted)
        base_height = 3072
        base_threshold = 5000
       
        # Get the dimensions of the image
        image_height, image_width, _ = image.shape

        # calculate scale factor. Works only if the image is scaled in comparison to the reference image and assuming uniform scaling in both dimensions
        scale = image_height / base_height

        # Initialize total coins value
        total_value = 0
        
        # Choose method of segmentation based on the 'kmeans' flag
        if kmeans:
            binary_image = kmeans_segmentation(image)
        else:
            binary_image = adaptive_segmentation(image)

        t2 = time()
        print(t2-t1, '!!')
        # Show the original and binary image
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.subplot(1, 2, 2)
        plt.title('Binary Image')
        plt.imshow(binary_image, cmap='gray')
        plt.show()

        # Find external contours in the binary image
        contours, hierarchy = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        if len(contours) == 0:
            print("No contours found in the image")
            continue
        
        # Initialize number of coins detected
        num_of_coins=0

        # Iterate through detected contours
        for contour in contours:
            
            # Fit a min enclosing circle only to contours of sufficient area. This is done in order to get rid of smaller contours and to keep
            # only the external perimeter contour of each coin

            # Dynamically adjusted threshold, works well only when the different scale image is a scaled version of the reference truth image
            if cv.contourArea(contour)> base_threshold*(scale**2): 

                # Fit the minimum enclosing circle and obtain the radius - descriptor of each coin
                (x, y), radius = cv.minEnclosingCircle(contour)

                # Draw a circle with the parameters (x,y,r)
                cv.circle(image,(int(x),int(y)), int(radius), (0,255,0), 9)

                # Increase number of coins detected by 1
                num_of_coins+=1
                
                # Compare the radius - descriptor of the detected coin to the known coin descriptors-dict values loaded through the JSON file
                # The value of the detected coin is equal to the corresponding dict key value of the descriptor that is closest to the radius of the coin
                closest_radius = None
                closest_key = None
                min_difference = float('inf')

                for key,descriptor in radiuses.items():
                    abs_diff = abs(descriptor-radius)
                    if abs_diff < min_difference:
                        min_difference = abs_diff
                        closest_radius = descriptor
                        closest_key = float(key)
                # Annotate the coin value
                #cv.putText(image, str(closest_key)+" euro", (int(x-radius/2), int(y+radius/2)), cv.FONT_HERSHEY_TRIPLEX, min(image_width,image_height)*0.001, (0,0,0), 
                #           math.ceil(min(image_width, image_height) * 0.001), cv.LINE_AA)
                print(closest_key)
                # Increase the total value of coins detected in the image
                total_value += closest_key

        # Annotate the number of coins detected, and the total value using dynamic font size and line width based on image dims
        cv.putText(image, f"Coins Detected: {num_of_coins}", (50,int(image_height/5)), cv.FONT_HERSHEY_TRIPLEX, min(image_width,image_height)*0.002, (0,0,0), 
                   math.ceil(min(image_width, image_height) * 0.001), cv.LINE_AA)
        cv.putText(image, f"Total Value: {total_value:.2f}"+" euro", (50,int(image_height/3)), cv.FONT_HERSHEY_TRIPLEX, min(image_width,image_height)*0.002, (0,0,0), 
                    math.ceil(min(image_width, image_height) * 0.001), cv.LINE_AA)
        # Show image
        plt.imshow(cv.cvtColor(image,cv.COLOR_BGR2RGB))
        plt.show()

detect_recognize_radiuses(images, radiuses, kmeans=True)
#detect_recognize_radiuses(scaled_images, radiuses, kmeans=False)
#detect_recognize_radiuses(brightness_adjusted_images, radiuses, kmeans=False)
#detect_recognize_radiuses(web_images, radiuses)



# Verified that scaling breaks the method !!!!!!
# Also the brightness changes broke it although it shouldnt. It is not the fault of the descriptor rather
# that the segmentation method used does not work correctly so the fitting of the min enclosing circles is wrong thus resulting
# in wrong detection and recognition
       
# the new method of detecting is more robust to ilummination changes, it detects the coins but the
# circles fitted radiuses are not always correct so the recognition is not ideal.  The dynamic nature of it deals with the changes in scale in some degree
# Testing on same scale and brightness images does not give completely ideal results. The segmentation is not the best
# resulting in not the best fitted circles so the very not robust descriptor of radiuses fails some times. 
# there are probably some minor changes in illumination due to the reflective background in regards to the original ground truth images
# of the same scale and brithness

# TODO: test other detection - segmentantion methods such as clustering


#### RESULTS COMBINED ####

# 1. For the adaptive thresholding and morphological filters segmentantion - detection algrotithm:

     # a. It detects with few errors  in general but it is not ideal resulting in not ideally fitted circles which results in 
     # some false recognitions due to the highly sensitive nature of the radiuses descriptor! Even at the ground truth images,
     # where we detect coins and extract the ground truth radiuses, the fits are not 100 percent correct due to shadows around the coins

     # b. We can see that when tested on same scale and brightness images. In this case we should have no errors but we do.
     # Most likely due to minor changes in brigthness due to the reflective background that result in not ideal circles fitted, 
     # some of the fits are different from the ground truth images which again these were also not perfect. So it seems they are both
     # inperfect but in different way resulting in some false recognitions.

     # c. When tested on the scaled images the detection part has similar outcomes as previously mentioned(not ideal fits), but this is not that visible
     # due to the dynamic change in line width which results in wider lines drawn. But the scaled images
     # are the scaled versions of the test images  mentioned above (b)|so the brightness levels are exacty the same so the same not ideal fit
     # should happen with the detection part. About the recognition part. It was expected that the algorithm would fail because
     # the radiuses descriptors are not robust to scale changes!

     # d. The brightness changes as mentioned at point c result in differences in the fit. The differences are not because the
     # radius drscriptor is not robst to brightness changes but becase the detection part is not robust to brightness changes
     # and the segmentantion through the adaptive thresholding yields different results in a degree, when the brightness changes.

     # Finally about the web images , these without occlusions. Some of the coins are not detected at all which is something
     # that so far has not happened to other images. This is because either the formula for dynamic adjustment of kernel size and number
     # of iterations in the morpological losing is not the best, OR MOST LIKELY AS IT SEEMS it has to do with the THRESHOLD VALUE FOR THE AREA OF CONTOURS !!!
     # which should ideally also be adjusted according to scale !!!!!

     # SO TO SUM UP, THIS DETECTION METHOD IS PRETTY GOOD BUT DOES NOT YIELD THE IDEAL FIT OF CIRCLES TO COINS WHICH IN TURN
     # HAS AN EFFECT ON THE RECOGNITION DUE TO THE SENSITIVE NATURE OF THE RADIUS DESCRIPTOR. WHEN THE DESCRIPTOR CHANGES, FOR
     # EXAMPLE WHEN WE WILL USE SIFT THAT SHOULD NOT AFFECT US, MEANING THE DETECTION WILL BE GOOD ENOUGH. !!! AS LONG AS WE HAVE 
     # A CONSTANT SMOOTH BACKGROUND (WHITE IN THIS CASE), AND NO OCLUSSIONS !!!!.


# A MORE ROBUST DETECTION ALGORITH COULD SOLVE SOME OF OUR PROBLEMS AND COUPLED WITH A MORE ROBUST DESCRIPTOR THE TOTAL
# PROGRAM SHOULD BECOME A LOT MORE RELIABLE !!!

### KMEANS DETECTION ###
# 1. The first thing to note is that the detecion takes a noticable more time to give an outome compared to the current detector.
  # (more smaller images this will not be a problem, but for high resolution images it is)
# 2. The segmentation is good but with the coins the the shadow around the coins is also detected which might also result 
# in not ideally fitted circles thus affecting again the sensitive radius descriptor. It wont have that affect in the SIFT descriptor

# It turns out the fitting is superior for this case resulting in far better fitted circles than the threshold detector and 
# because of that fewer misclasified coins
 
# 3. We need to test to see if it is more robust to illumination and scaling changes compared to the current detector.

    # a. First the kmeans detector used with the radiues descriptor made no error on the test images, where as the adaptive threhold 
    # detector made a number of errors . It seems like despite the shadows captured the detection is far better resulting in no mistakes made
    # on the test images.

    # b. Testing on the scaled images cannot act as comparison measure between the two detectors. 
    #  because the descriptor will fail both times so both times the results will be completely wrong. The comparison on scaling
    # can be tested when using SIFT descriptors which are scale invariant, then any change will be due to the detector's nature

    # c. So lets test on the brightness changed images. The kmeans detector made again no mistakes, compared to the threshold detector
    # which not only made mistakes but it was actually not robust to illumination changes giving different results compared to the
    # the ones that it gave for the original test images.

    # d. When tested on the web images the result are again better for the detection using Kmenas compared to the current.
    # It detected some coins that the current misses. The thing is that while the Kmeans decreases the number of parameters used
    # there is still the threshold value paramater for the contour area. The base threshold 5000 is dynamically adjusted by 
    # the scale ^ 2. It seems to work better, it wins some, it loses some. Further testing should be done. For now we will use 
    # the dynamicaly adjusted version