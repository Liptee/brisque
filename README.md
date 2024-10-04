# BRISQUE Image Quality Assessment

This repository provides a Python implementation for training and calculating the **Blid/Referenceless Image Statial Quality Evaluator** (BRISQUE) metric. The code allows you to compute BRISQUE features, train a **Support Vector Regression** (SVR) model using these features and **Mean Opinion Scores** (MOS), and predict BRISQUE scores for new images.

## Introduction
**BRISQUE** is a no-reference image quality assessment algorithm that measrues the naturalness of an image without any reference image. It analyzes the image in the spatial domain using **natural scene statisctics** (NSS) and models these statistics to quantify possible losses of "naturalness" due to distortion.

This implementation allows you to:
- compute BRISQUE features from images
- train an SVR model using computed features and MOS
- predict BRISQUE scores for new images using the trained model

## Requirements
- Python 3.8 (recommended)
- NumPy
- SciPy
- scikit-learn
- opencv-python

Install requirements from requirements.txt or using this command:
```bash
pip install numpy scipy scikit-learn opencv-python
```

## Usage
### Fast test
To be sure that's working you can use pretrained `pkl` file:
```python
from brisque import brisque
import joblib

model = joblib.load('svm_pkls/tid2008.pkl')

image_path = 'images/Lenna.png'
score = brisque(image_path, model)
print(f'BRISQUE Score: {score}')
```


### Training the SVR Model
To train an SVR model using BRISQUE features and corresponding MOS values:
```python
from brisque import train_brisque_model

# List of image paths and corresponding MOS values
image_paths = ["image1.jpg", "image2.jpg", ...]
mos_list = [mos_value1, mos_value2, ...]

# Train the SVR model
model = train_brisque_model(image_paths, mos_list)
```

### Predicting BRISQUE Score
To predict the BRISQUE score for a new image:
```python
from brisque import brisque

image_path = "path_to_your_image.png"

score = brisque(image_path, model)

print(f"Predicted BRISQUE score: {score}")
```

## Function Documentation
#### `estimate_ggd_parameters(vec)`
Estimate the parameters of a Generalized Gaussian Distribution (GGD) for the input vector.

Parameters:
- vec (numpy.ndarray): Input vector.

Returns:
- alpha (float): Shape parameter.
- sigma (float): Standard deviation.

#### `estimate_aggd_parameters(vec)`
Estimate the parameters of an Asymmetric Generalized Gaussian Distribution (AGGD) for the input vector.

Parameters:
- vec (numpy.ndarray): Input vector.

Returns:
- alpha (float): Shape parameter.
- left_std (float): Left standard deviation.
- right_std (float): Right standard deviation.

#### `compute_brisque_features(img)`
Compute BRISQUE features for a given image.

Parameters:
- img (numpy.ndarray): Input image in RGB or grayscale format.

Returns:
- features (list): Extracted BRISQUE features (length 36).

#### `compute_dataset_features(image_paths)`
Compute BRISQUE features for a list of images.

Parameters:
- image_paths (list): List of image paths.

Returns:
- features_list (list): List of feature vectors.

#### `train_svm_model(features_list, mos_list)`
Train an SVR model using BRISQUE features and MOS values.

Parameters:
- features_list (list): List of feature vectors.
- mos_list (list): Corresponding Mean Opinion Scores.

Returns:
- model (sklearn.pipeline.Pipeline): Trained SVR model.

#### `replace_infinite_values(features_list)`
Replace infinite and NaN values in the features list with finite numbers.

Parameters:
- features_list (list): List of feature vectors.

Returns:
- features_list_clean (list): Cleaned list of feature vectors.

#### `brisque_score(img, model)`
Predict the BRISQUE quality score for an image.

Parameters:
- img (numpy.ndarray): Input image in RGB or grayscale format.
- model (sklearn.pipeline.Pipeline): Trained SVR model.

Returns:
- score (float): Predicted BRISQUE score.

#### `brisque(path_to_img, model)`
Predict the BRISQUE quality score for an image given its file path.

Parameters:
- path_to_img (str): Path to the target image.
- model (sklearn.pipeline.Pipeline): Trained SVR model.

Returns:
- score (float): Predicted BRISQUE score.

#### `train_brisque_model(image_paths, mos_list)`
Train an SVR model using BRISQUE features extracted from images and their corresponding MOS values.

Parameters:
- image_paths (list): List of image paths.
- mos_list (list): Corresponding Mean Opinion Scores.

Returns:
- model (sklearn.pipeline.Pipeline): Trained SVR model.

## Notes
1. Image Preprocessing: Ensure that images are properly loaded and converted to the RGB color space if they are read using OpenCV, which loads images in BGR format by default.
2. Model Parameters: The SVR model is configured with an RBF kernel, C=200, and gamma=0.0001. You may need to adjust these hyperparameters based on your specific dataset.
3. Data Cleaning: The code includes functions to handle infinite and NaN values in the feature vectors. It’s important to clean the features to avoid issues during model training and prediction.
