# Too_Lazy_to_Count-TLtC-A_coin_detection_and_recognition_project
Detecting and Recognizing Coins utilizing classic Image Processing and Computer Vision techniques.

With this project, I aim to tackle the problem of detecting coins in images, recognizing their respective values, and calculating the total value of coins present utilizing classic computer vision and image processing methodologies. There are two main aspects to this problem. First, a coin detection algorithm is necessary to detect the coins that appear in an image. Secondly, the program should be able to recognize the type of each detected coin, and to do so, a coin descriptor and a matching method must be implemented.

This repository contains 5 files. This README.md file, a pfd file containing a detailed project report and 4 .py files. In detail, these .py files are the following:

1) main.py : Contains the code that extracts the descriptors from the set of ground truth images, creating two different databases (JSON files) of reference values.
2) image_augmentation.py : Contains the code that performs image augmentation (scaling, brightness transformation) and saves the output images.
3) radius_descriptors.py : Contains the code that performs coin detection ( the detection method is user defined) and recognition through radius descriptor matching.
4) SIFT_descriptors.py : Contains the code that performs coin detection (the detection method is user defined) and recognition through SIFT descriptors matching.

Finally, the ground truth images (used for extracting the coin descriptions) and the test images are uploaded.
