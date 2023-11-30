from elice_utils import EliceUtils
from generate import sampling
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
import cv2
elice_utils = EliceUtils()

def plt_show():
    plt.savefig("fig")
    elice_utils.send_image("fig.png")


def construct_matrix_A(src_points, dst_points):
    """
    Construct the matrix A using source and destination points.
    src_points: (4,2) 
    dst_points: (4,2)
    """
    #TODO Derive the 9x2 A matrix. What should be included in the A matrix?
    A=[]
    for src,dst in zip(src_points,dst_points):
        x1,y1=src
        x2,y2=dst
        # A should be like an 8x9 matrix, no?
        A1=[-x1,-y1,-1,0,0,0,x2*x1,x2*y1,x2]
        A2=[0,0,0,-x1,-y1,-1,y2*x1,y2*y1,y2]
        A.append(A1)
        A.append(A2)
    return A

def compute_homography(src_points, dst_points):
    """
    Compute the homography matrix using SVD.
    Your answer should involve A matrix. 
    """
    #TODO
    A=construct_matrix_A(src_points,dst_points)
    U,s,VT = svd(A)
    H = VT[-1,:].reshape((3,3))
    # For normalizing w.r.t the scale vector
    H = H / H[-1, -1]
    return H

def reconstruct_image(image,H):
    """
    Reconstruct warped image back to original using cv2 library.
    Hint: Use cv2.warpPerspective
    """
    reconstructed_image=cv2.warpPerspective(image,np.linalg.inv(H),(image.shape[1],image.shape[0]))
    return reconstructed_image

def main():
    image1, image2, src, dst = sampling('macao.jpg')
    H = compute_homography(src,dst)
    reconstructed_image = reconstruct_image(image1,H)


if __name__ == "__main__":
    main()
