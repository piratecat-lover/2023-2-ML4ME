from elice_utils import EliceUtils
import matplotlib.pyplot as plt
import numpy as np
import cv2

elice_utils = EliceUtils()

def plt_show():
    plt.savefig("fig")
    elice_utils.send_image("fig.png")


def visualize_histogram_equalization(img):
    # Load the image and convert to RGB
    # Apply the histogram equalization function
    equalized_data = histogram_equalization(img.tolist())
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Original grayscale image
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Original Grayscale Image")
    ax[0].axis('off')

    # Histogram equalized image
    ax[1].imshow(equalized_data, cmap='gray')
    ax[1].set_title("Histogram Equalized Image")
    ax[1].axis('off')

    plt.tight_layout()
    plt_show()



def compute_histogram(img):
    """
    Computes the histogram of a grayscale image.

    Input:
    - img: A 2D numpy array representing a grayscale image with pixel values in [0, 255].

    Returns:
    - histogram: A list of length 256 where histogram[i] is the count of pixels with intensity value i.
    """
    # TODO: Implement this function.
    img=np.array(img)
    flattened_img=img.flatten()
    output=list(np.bincount(flattened_img))
    if len(output)<256: output.extend([0]*(256-len(output)))
    # output=cv2.calcHist(img,[0],None,[256],[0,256])
    return output

def compute_CDF(histogram):
    """
    Computes the Cumulative Distribution Function (CDF) based on a given histogram.

    Input:
    - histogram: A list of length 256 representing the histogram of a grayscale image.

    Returns:
    - cdf: A list of length 256 representing the cumulative sum of the histogram.
    """
    # TODO: Implement this function.
    output=np.cumsum(histogram)
    output=list(output)
    return output

def normalize_CDF(img, cdf):
    """
    Normalizes a given CDF to the range [0, 255].

    Inputs:
    - img: A 2D numpy array representing a grayscale image.
    - cdf: A list of length 256 representing the CDF of the image.

    Returns:
    - normalized_cdf: A list of length 256 where each value is scaled to the range [0, 255].
    """
    # TODO: Implement this function.
    normalized_img=cv2.equalizeHist(img)
    norm_hist=compute_histogram(normalized_img)
    output=np.cumsum(norm_hist)
    output=list(output)
    return output

def remap_pixels(img, cdf):
    """
    Remap each 
    Inputs:
    - img: A 2D numpy array representing a grayscale image.
    - cdf: A list of length 256 representing the CDF of the image.

    Returns:
    - equalized_image: 2D list of length each item in the list is the rescaled pixel value to the image between [0,255].
    """
    equalized_img=cv2.equalizeHist(img)
    equalized_img=np.array(equalized_img)
    return equalized_img


def histogram_equalization(img):
    """
    Equalizes the histogram of a grayscale image.

    Input:
    - img: A 2D numpy array representing a grayscale image with pixel values in [0, 255].

    Returns:
    - equalized_img: A 2D list representing the histogram-equalized image.
    """
    img=np.array(img)
    img=img.astype(np.uint8)
    # Step 1: Compute the histogram
    hist1=compute_histogram(img)
    # Step 2: Compute CDF
    cdf1=compute_CDF(hist1)
    # Step 3: Normalize CDF
    norm_cdf=normalize_CDF(img, cdf1)
    # Step 4: Map the original intensity values to new values
    mapped_img=remap_pixels(img,norm_cdf)
    return mapped_img



def main():
    img_path = "./img1.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
   
    # 3. Apply histogram equalization using the function provided earlier

    visualize_histogram_equalization(gray_img)

   
    
    elice_utils.send_image('elice.png')
    elice_utils.send_file('data/input.txt')



if __name__ == "__main__":
    main()
