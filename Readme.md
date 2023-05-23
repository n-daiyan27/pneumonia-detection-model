# Pneumonia Detection Using Chest X-ray with Using Image Processing and Deep Learning


# Dataset

The dataset was obtained from Kaggle and contains 5,856 images in total, including 3,856 images of patients with pneumonia and 2,000 normal chest X-ray images.

# Model Architecture

The model uses a convolutional neural network (CNN) architecture, specifically the VGG16 model, which has been pre-trained on the ImageNet dataset.

# Modified dataset

I have tried to do some image processing on the dataset. I have applied clahe on the chest x-ray images to get the better result and the contouring the images of training dataset. Later used that modified training dataset to build the model for detecting pneumonia

# How to run
To use the pneumonia detection model, you will need to have TensorFlow installed on your machine. You can then clone the repository and run the model.py script, passing in the path to the chest X-ray image that you want to train.
  python model.py
The script will train the datasets and generate a h5 model.
