import cv2
import numpy as np

def extract_image_features(image):
    """
    Function to extract features from image data.

    Parameters:
        image (array): The raw image array in grayscale or color.

    Returns:
        vector (array): The extracted feature vector.
    """

    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resizing the image to a standard size (128x64)
    image = cv2.resize(image, (128, 64))

    # Initialize HOG descriptor
    hog = cv2.HOGDescriptor()

    # Compute HOG features
    hog_features = hog.compute(image)

    # Flatten the array for use as a feature vector
    vector = hog_features.flatten()

    return vector
