from elice_utils import EliceUtils
import numpy as np
from generate import generate_data
import matplotlib.pyplot as plt
from scipy.stats import norm
elice_utils = EliceUtils()

def plt_show():
    plt.savefig("fig")
    elice_utils.send_image("fig.png")

def gaussian_pdf(x, mu, sigma):
    """
    Compute Gaussian pdf. (Refer to lecture)
    """
    #TODO
    pdf=1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*(sigma**2)))
    # pdf=norm.pdf(x,mu,sigma) #HAHAHAHAHAHA
    return pdf

def estimate_robot_position(landmarks, noisy_distances, sigma, prior_mean, prior_variance):
    """Estimate robot position using Maximum A Posteriori (MAP) estimation. If you are not sure, follow the pseudocode in the instructions. 
    """
    # Create an array of possible positions
    landmarks=np.asarray(landmarks)
    noisy_distances=np.asarray(noisy_distances)
    prior_mean=np.asarray(prior_mean)
    positions = np.array([[x, y] for x in range(100) for y in range(100)])
    max_posterior = float('-inf')
    best_position = None
    for pos in positions:
        likelihood = 1
        #TODO: Think about how MLE and MAP can be calculated by finding the expected values?
        exp_dist=np.linalg.norm(landmarks-pos,axis=1)
        prior_dist=np.linalg.norm(landmarks-prior_mean,axis=1)
        for i,dist in enumerate(exp_dist):
            likelihood*=gaussian_pdf(noisy_distances[i],dist,sigma)*gaussian_pdf(dist,prior_dist[i],prior_variance)
        if likelihood>max_posterior:
            max_posterior=likelihood
            best_position=pos
    return tuple(best_position)
    
def visualize(true_position, estimated_position, landmarks):
    """Visualize the true position, estimated position, and landmarks."""
    plt.scatter(*zip(*landmarks), marker='o', color='red', label='Landmarks')
    plt.scatter(*true_position, marker='x', color='blue', label='True Position')
    plt.scatter(*estimated_position, marker='x', color='green', label='Estimated Position')
    plt.legend()
    plt.grid(True)
    plt_show()


def main():
    true_position = (45,45)
    landmarks = [(10, 10), (90, 10), (90, 90), (10, 90)]
    sigma = 5
    prior_mean = (50, 50)
    prior_variance = 100

    # Generate data
    noisy_distances = generate_data(true_position, landmarks, sigma, prior_mean, prior_variance)

    # Estimate robot position using MAP
    estimated_position = estimate_robot_position(landmarks, noisy_distances, sigma, prior_mean, prior_variance)

    # Visualize
    visualize(true_position, estimated_position, landmarks)


if __name__ == "__main__":
    main()









