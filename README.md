# Too_Lazy_to_Count-TLtC-A_coin_detection_and_recognition_project
Detecting and Recognizing Coins utilizing classic Image Processing and Computer Vision techniques.
![image](https://github.com/user-attachments/assets/4268c859-142a-4b92-9185-f3cf4c1d42a8)



With this project, I aim to tackle the problem of detecting coins in images, recognizing their respective values, and calculating the total value of coins present utilizing classic computer vision and image processing methodologies. There are two main aspects to this problem. First, a coin detection algorithm is necessary to detect the coins that appear in an image. Secondly, the program should be able to recognize the type of each detected coin, and to do so, a coin descriptor and a matching method must be implemented. The code implementation is in Python mainly utilizing the modules NumPy and OpenCV.

This repository contains 6 files. This README.md file, a pfd file containing a detailed project report and 4 .py files. In detail, these .py files are the following:

1) main.py : Contains the code that extracts the descriptors from the set of ground truth images, creating two different databases (JSON files) of reference values.
2) image_augmentation.py : Contains the code that performs image augmentation (scaling, brightness transformation) and saves the output images.
3) radius_descriptors.py : Contains the code that performs coin detection ( the detection method is user defined) and recognition through radius descriptor matching.
4) SIFT_descriptors.py : Contains the code that performs coin detection (the detection method is user defined) and recognition through SIFT descriptors matching.

Finally, the ground truth images (used for extracting the coin descriptors) and the test images are uploaded.



*** In page 11 of the report the dynamically adjusted parameters of the program are discussed. I failed to mention that the contour area threshold (which is one of them) is not limited just to the adaptive thresholding method of detection. It is independent of the detection method and it is a variable used also in the case where the detection method is based on KMeans clustering. That being said, I must note that changes in scale might affect the detection results provided by the KMeans method due to how this threshold value is calculated based on the scaling factor value. ***
