import cv2
import numpy as np
from typing import List

from utils import *


def brisque(path_to_img, model):
    """
    Predict the BRISQUE quality score for an image.

    Parameters:
        path_to_img (string): Path to target image.
        model (sklearn.pipeline.Pipeline): Trained SVR model.

    Returns:
        score (float): Predicted BRISQUE score.
    """
    img = cv2.imread(path_to_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    score = brisque_score(img, model)
    return score


def train_brisque_model(image_paths: List, mos_list: List):
    """
    Train an SVR model using BRISQUE features and MOS.

    Parameters:
        image_paths (list): List of image paths.
        mos_list (list): Corresponding Mean Opinion Scores.

    Returns:
        model (sklearn.pipeline.Pipeline): Trained SVR model.
    """
    features_list = compute_dataset_features(image_paths)
    features_list = replace_infinite_values(features_list)

    max_float = np.finfo(np.float64).max
    median_value = np.median(features_list, axis=0)
    features_list = np.where(features_list == max_float, median_value, features_list)

    model = train_svm_model(features_list, mos_list)
    return model
