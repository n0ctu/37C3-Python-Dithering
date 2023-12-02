import numpy as np
import cv2

def dithering(image, dithering_algorithm='burkes'):

    # Prepare the image in grayscale and the dimensions
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = gray_image.shape
    dithered_image = np.copy(gray_image)

    # The error distribution matrixes (according to: https://tannerhelland.com/2012/12/28/dithering-eleven-algorithms-source-code.html)
    if dithering_algorithm == 'floyd-steinberg':
        error_matrix = np.array([[0, 0, 7],
                                 [4, 5, 1]]) / 16
        
    if dithering_algorithm == 'burkes':
        error_matrix = np.array([[0, 0, 0, 8, 4],
                                 [2, 4, 8, 4, 2]]) / 32
    elif dithering_algorithm == 'sierra':
        error_matrix = np.array([[0, 0, 0, 5, 3],
                                 [2, 4, 5, 4, 2], 
                                 [0, 2, 3, 2, 0]]) / 32

    elif dithering_algorithm == 'sierra-two-row':
        error_matrix = np.array([[0, 0, 0, 4, 3],
                                 [1, 2, 3, 2, 1]]) / 16
        
    elif dithering_algorithm == 'sierra-lite':
        error_matrix = np.array([[0, 0, 2],
                                 [1, 1, 0]]) / 4
        
    # Loop through each pixel in the image
    for y in range(rows):
        for x in range(cols):
            # Get the original pixel value
            old_pixel = dithered_image[y, x]
            # New pixel value (black or white)
            new_pixel = 255 if old_pixel > 127 else 0
            dithered_image[y, x] = new_pixel
            # Calculate the quantization error
            error = old_pixel - new_pixel

            # Distribute the error to neighboring pixels
            for dy in range(error_matrix.shape[0]):
                for dx in range(error_matrix.shape[1]):
                    # Move cursor to the center of the first row of the error matrix (starting point)
                    new_x = x + dx - (error_matrix.shape[1] // 2)
                    new_y = y + dy

                    # Fairly inefficient ...
                    if new_x >= 0 and new_x < cols and new_y >= 0 and new_y < rows:
                        dithered_image[new_y, new_x] += error * error_matrix[dy, dx]

    return dithered_image

# Load an image
image = cv2.imread('lenna.jpg')

# Apply dithering
dithered_image = dithering(image, 'burkes')

# Display the dithered image for now
cv2.imshow('Dithered Image', dithered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()