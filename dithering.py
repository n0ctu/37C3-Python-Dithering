import numpy as np
import cv2
from PIL import Image

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
            for erry in range(error_matrix.shape[0]):
                for errx in range(error_matrix.shape[1]):

                    new_x = x + errx - (error_matrix.shape[0] // 2)
                    new_y = y + erry

                    # Skip if out of bounds
                    if new_x >= 0 and new_x < cols and new_y >= 0 and new_y < rows:
                        updated_pixel = error * error_matrix[erry, errx]
                        # Added clipping
                        dithered_image[new_y, new_x] = np.clip(dithered_image[new_y, new_x] + updated_pixel, 0, 255)

        cv2.imshow('Dithered Image', dithered_image)
        cv2.waitKey(10)

    return dithered_image

def video_to_frames(video_path, max_frames=None):
    video = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while video.isOpened() and (max_frames is None or frame_count < max_frames):
        ret, frame = video.read()
        if ret:
            frames.append(frame)
            frame_count += 1
        else:
            break

    video.release()
    return frames

def frames_to_gif(frames, gif_path, fps=24):
    pil_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], loop=0, duration=1000/fps)


def main():
    
    input = 'input.mp4'
    width = 256
    height = 256
    fps = 24
    max_frames = None
    dither_algorithm = 'floyd-steinberg'

    # Read the video and convert it to frames
    frames = video_to_frames(input, max_frames)

    
    # Dither each frame
    dithered_frames = []
    for frame in frames:
        # Resize the frame to the desired dimensions
        frame = cv2.resize(frame, (width, height))
        dithered_frame = dithering(frame, dither_algorithm)
        dithered_frames.append(dithered_frame)
        print("Processing: Frame", len(dithered_frames), "of", len(frames))

    # Save the frames as a GIF
    frames_to_gif(dithered_frames, 'video.gif', fps)

    '''
    # Single Image Testing
    image = cv2.imread('test.jpg')
    resized_image = cv2.resize(image, (256, 256))
    dithered_image = dithering(resized_image, 'burkes')

    # Display the dithered image
    cv2.imshow('Dithered Image', dithered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

if __name__ == '__main__':
    main()