# External Imports
import glob
import cv2
import keras
import os
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Flatten, Dense, Input

class DataLoader(keras.utils.Sequence):
    def __init__(self, directory: os.PathLike, subset: Literal["training", "validation"], test_split: float, seed: int, batch_size: int, image_size: tuple[int, int], bins = None, bin_threshold = 200):
        """
        Loads data from path and the data is transformed when the batch is called
        :param directory: The outermost path that stores the separate passes of recorded data
        :param subset: Which subset to load
        :param test_split: percentage of test data to split off
        :param seed: Random seed
        :param batch_size: The size of the batch
        :param image_size: The size of the image to be resized to
        :param bins: The number of bins to use for binarization
        :param bin_threshold: The threshold for binarization
        """
        super().__init__()
        self.path = directory
        self.subset = subset
        self.data = pd.DataFrame()
        self.batch_size = batch_size
        self.image_size = image_size
        self.length = 0
        self.seed = seed
        self.bins = bins

        columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
        # Loop through directories and concat
        for csv_file in glob.glob(os.path.join(directory, "*/*.csv")):
            df = pd.read_csv(csv_file, names=columns)
            df = df[["center", "steering", "throttle", "brake", "speed"]]

            self.data = pd.concat([self.data, df], ignore_index=True)

        # Create Bins if none entered
        if self.bins is None:
            steering_values_max = max(self.data["steering"])
            steering_values_min = min(self.data["steering"])
            self.bins = np.arange(steering_values_min, steering_values_max, 0.05)

        # Add binning to dataframe as a column
        steering_binned = np.digitize(self.data["steering"], self.bins)
        self.data["steering_bin"] = steering_binned

        # Shuffle the dataset
        self.data = self.data.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        # Split the dataset based on the split amount
        train, test = train_test_split(self.data, test_size=test_split, random_state=self.seed)

        if self.subset == "training":
            # Balance based on binning
            bin_counter = {i: 0 for i in range(len(self.bins) + 1)}
            idx_to_add = []

            # Shuffle then save index to add if the bin count is less than the threshold
            for row in train.itertuples(index=True, name='Pandas'):
                if bin_counter[row.steering_bin] < bin_threshold:
                    idx_to_add.append(row.Index)
                    bin_counter[row.steering_bin] += 1

            self.data = train.loc[idx_to_add]

        elif self.subset == "validation":
            self.data = test

    def bin_count(self):
        return {i: len(self.data[self.data["steering_bin"] == i]) for i in range(0, len(self.bins) + 1)}

    def visualize_bins(self):
        import matplotlib.pyplot as plt
        bin_counts = self.bin_count()
        plt.bar(bin_counts.keys(), bin_counts.values())
        plt.xlabel("Steering Bin Count")
        plt.ylabel("Count")
        plt.title("Distribution of Steering Bins")
        plt.show()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_data = self.data.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        steering_angles = []

        for image, steering in zip(batch_data["center"], batch_data["steering"]):
            # Basic preprocessing
            image = cv2.imread(image)                                       # Load image
            image = image[60:135, :, :]                                     # Crop the image to remove the sky and the hood of the car
            image = cv2.resize(image, self.image_size)                      # Resize image
            image = cv2.GaussianBlur(image, (5, 5), 0)        # Gaussian Blur

            # Augments
            if self.subset == "training":
                # Randomly flip the image (Over Vertical)
                if np.random.random() < 0.5:
                    image = cv2.flip(image, 1)
                    steering = -steering                        # If the image was flipped, flip the steering angle as well

                # Random Brightness / Contrast
                if np.random.random() < 0.5:
                    brightness = np.random.randint(-20, 20)
                    contrast = np.random.uniform(0.8, 1.2)
                    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

                # Random Panning
                if np.random.random() < 0.5:
                    tx = np.random.uniform(-0.1,0.1) * image.shape[1]
                    ty = np.random.uniform(-0.1,0.1) * image.shape[0]
                    M = np.float32([[1, 0, tx], [0, 1, ty]])
                    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

                # Random Rotation & Zoom
                if np.random.random() < 0.5:
                    angle = np.random.uniform(-5, 5)
                    scale = np.random.uniform(1.0, 1.1)
                    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, scale)
                    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)      # Convert to YUV color space
            image = image / 255.0                               # Normalize pixel values to [0, 1]

            images.append(image)
            steering_angles.append(steering)

        return np.array(images), np.array(steering_angles)


NvidiaCNN = keras.Sequential([
    Input(shape=(66, 200, 3)),
    Conv2D(filters=24, kernel_size=(5, 5), strides=2, activation="relu", padding="valid"),
    Conv2D(filters=36, kernel_size=(5, 5), strides=2, activation="relu", padding="valid"),
    Conv2D(filters=48, kernel_size=(5, 5), strides=2, activation="relu", padding="valid"),
    Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu", padding="valid"),
    Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu", padding="valid"),
    Flatten(),
    Dense(1164, activation="relu"),
    Dense(100, activation="relu"),
    Dense(50, activation="relu"),
    Dense(10, activation="relu"),
    Dense(1)
])