from scipy.special import gamma
import numpy as np
from typing import List, Tuple
import cv2
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def estimate_ggd_parameters(vec: np.ndarray) -> Tuple[float, float]:
    """
    Estimate the parameters of a Generalized Gaussian Distribution (GGD).

    Parameters:
        vec (numpy.ndarray): Input vector.

    Returns:
        alpha (float): Shape parameter.
        sigma (float): Standard deviation.
    """
    gam = np.arange(0.2, 10, 0.001)
    r_gam = (gamma(1/gam) * gamma(3/gam)) / (gamma(2/gam) ** 2)

    sigma_sq = np.mean(vec ** 2)
    E = np.mean(np.abs(vec))
    rho = sigma_sq / (E ** 2)
    diff = np.abs(rho - r_gam)
    alpha = gam[np.argmin(diff)]
    sigma = E / (gamma(1/alpha) / gamma(2/alpha)) ** 0.5

    return alpha, sigma


def estimate_aggd_parameters(vec):
    """
    Estimate the parameters of an Asymmetric Generalized Gaussian Distribution (AGGD).

    Parameters:
        vec (numpy.ndarray): Input vector.

    Returns:
        alpha (float): Shape parameter.
        left_std (float): Left standard deviation.
        right_std (float): Right standard deviation.
    """
    left_vec = vec[vec < 0].astype(np.float64)
    right_vec = vec[vec > 0].astype(np.float64)

    left_std = np.sqrt((left_vec ** 2).mean()) if left_vec.size > 0 else 0
    right_std = np.sqrt((right_vec ** 2).mean()) if right_vec.size > 0 else 0

    if left_std == 0 or right_std == 0:
        return 0.1, 0, 0

    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(vec))) ** 2 / np.mean(vec ** 2)
    rhatnorm = rhat * (gammahat ** 3 + 1) * (gammahat + 1) / ((gammahat ** 2 + 1) ** 2)

    gam = np.arange(0.2, 10, 0.001)
    prec_gam = gamma(2/gam) ** 2 / (gamma(1/gam) * gamma(3/gam))
    diff = np.abs(prec_gam - rhatnorm)
    alpha = gam[np.argmin(diff)]

    const = np.sqrt(gamma(1/alpha) / gamma(3/alpha))
    left_std = left_std / const
    right_std = right_std / const

    return alpha, left_std, right_std


def compute_brisque_features(img: np.ndarray) -> List:
    """
    Compute BRISQUE features for a given image.

    Parameters:
        img (numpy.ndarray): input image in RGB or grayscale format

    Returns:
        features (list): Extracted BRISQUE features (length 36)
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32) / 255.0

    kernel_size = 7
    sigma = 7 / 6.0
    mu = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    mu_sq = mu * mu
    sigma = cv2.GaussianBlur(img * img, (kernel_size, kernel_size), sigma)
    sigma = np.sqrt(np.abs(sigma - mu_sq))

    sigma[sigma < 1/255.0] = 1/255.0
    mscn = (img - mu) / sigma

    # check mscn is valid value

    shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    features = []

    alpha_mscn, sigma_mscn = estimate_ggd_parameters(mscn)
    features.extend([alpha_mscn, sigma_mscn ** 2])

    for shift in shifts:
        shifted_mscn = np.roll(mscn, shift, axis=(0, 1))
        epsilon = 1e-10
        pair = np.power(np.abs(mscn) + epsilon, shifted_mscn)
        alpha, left_std, right_std = estimate_aggd_parameters(pair)
        const = (gamma(2/alpha) / gamma(1/alpha)) ** 0.5
        mean = (right_std - left_std) * (gamma(2/alpha) / gamma(1/alpha)) * const
        features.extend([alpha, (left_std ** 2), (right_std ** 2), mean])

    return features


def compute_dataset_features(image_paths: List) -> List:
    """
    Compute BRISQUE features for a list of images.

    Parameters:
        image_paths (list): List of image paths.

    Returns:
        features_list (list): List of feature vectors.
    """

    features_list = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        features = compute_brisque_features(img)
        features_list.append(features)

    return features_list


def train_svm_model(features_list: List, mos_list: List):
    """
    Train an SVR model using BRISQUE features and MOS.

    Parameters:
        features_list (list): List of feature vectors.
        mos_list (list): Corresponding Mean Opinion Scores.

    Returns:
        model (sklearn.pipeline.Pipeline): Trained SVR model.
    """
    pipeline = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=200, gamma=0.0001))
    pipeline.fit(features_list, mos_list)
    return pipeline


def replace_infinite_values(features_list):
    max_value = np.finfo(np.float64).max
    features_list_clean = []

    for features in features_list:
        features = np.where(np.isinf(features), max_value, features)
        features = np.where(np.isnan(features), 0, features)
        features_list_clean.append(list(features))

    return features_list_clean


def brisque_score(img, model):
    """
    Predict the BRISQUE quality score for an image.

    Parameters:
        img (numpy.ndarray): Input image in BGR or grayscale format.
        model (sklearn.pipeline.Pipeline): Trained SVR model.

    Returns:
        score (float): Predicted BRISQUE score.
    """
    max_float = np.finfo(np.float64).max

    features = compute_brisque_features(img)
    median_value = np.median(features, axis=0)
    features = np.where(np.isinf(features), max_float, features)
    features = np.where(np.isnan(features), 0, features)
    features = np.where(features == max_float, median_value, features)
    score = model.predict([features])[0]
    return score
