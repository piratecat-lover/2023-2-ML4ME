from elice_utils import EliceUtils
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
from dataset import MyDataset
import cv2

elice_utils = EliceUtils()


def plt_show():
    plt.savefig("fig")
    elice_utils.send_image("fig.png")

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # define necessary initial schemes here. E.g. hyperparameters/model loading etc..  
    def forward(self, image):
        # image = input image with dimension H x W (2 dimensional)
        # Prediction format must be written in prediction = [confidence, x1, y1, x2, y2]
        return predictions

def main():
    
    # For sanity checking model on test image. 
    
    model = MyModel()
    test_img = cv2.imread('./test_data/test.jpg',-1)
    #test_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    predictions = model(test_img)
    
    #print('predictions are=', predictions)
    # Convert image to RGB for displaying
    if len(test_img.shape) == 2:  # If grayscale
        test_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    elif test_img.shape[2] == 1:  # If single channel
        test_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)

    # Draw the bounding boxes and confidence on the image
    for prediction in predictions:
        confidence, x, y, w, h = prediction  # Unpack the tuple
        print('predictions are=', x,y,w+x, h+y)
        if confidence > 0:
            test_img = cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(test_img, f'Conf: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # Convert BGR image to RGB for plotting
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    plt.imshow(test_img_rgb)
    plt.axis('off')  # Hide axis
    plt_show()


if __name__ == "__main__":
    main()
