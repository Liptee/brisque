import joblib
from brisque import brisque

image_path = "images/Lenna.png"
model = joblib.load('brisque_model.pkl')

score = brisque(image_path, model)
print(f'BRISQUE Score: {score}')
