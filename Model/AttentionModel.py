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
        dataframes = []
        # Loop through directories and concat
        for csv_file in glob.glob(os.path.join(directory, "*/*.csv")):
            df = pd.read_csv(csv_file, names=columns)
            df = df[["center", "steering", "throttle"]]

            dataframes.append(df)


        # Build Windows
        all_windows = []
        for df in dataframes:

            for i in range(4, len(df)):
                frame_0 = df.iloc[i - 4]["center"]         # oldest
                frame_1 = df.iloc[i - 3]["center"]
                frame_2 = df.iloc[i - 2]["center"]
                frame_3 = df.iloc[i - 1]["center"]
                frame_4 = df.iloc[i]["center"]             # newest / current
                steering = df.iloc[i]["steering"]
                window = {
                    "frame_0": frame_0,
                    "frame_1": frame_1,
                    "frame_2": frame_2,
                    "frame_3": frame_3,
                    "frame_4": frame_4,
                    "steering": steering
                }

                all_windows.append(window)

        self.data = pd.DataFrame(all_windows)

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
        windows = []
        steering_angles = []

        for row in batch_data.itertuples(index=True, name='Pandas'):
            frames = [row.frame_0, row.frame_1, row.frame_2, row.frame_3, row.frame_4]
            steering = row.steering

            flip = np.random.choice([True, False])

            brightness = np.random.choice([True, False])
            alpha_contrast = np.random.uniform(0.8, 1.2)
            beta_brightness = np.random.randint(-20, 20)

            panning = np.random.choice([True, False])
            tx = np.random.uniform(-0.1, 0.1)
            ty = np.random.uniform(-0.1, 0.1)

            rotation = np.random.choice([True, False])
            angle = np.random.uniform(-5, 5)
            scale = np.random.uniform(1.0, 1.1)

            if flip:
                steering = -steering  # If the image was flipped, flip the steering angle as well - Image flip happens in frame processing

            processed_frames = []
            for frame in frames:
                # Basic preprocessing
                frame = cv2.imread(frame)                                       # Load image
                frame = frame[60:135, :, :]                                     # Crop the image to remove the sky and the hood of the car
                frame = cv2.resize(frame, self.image_size)                      # Resize image
                frame = cv2.GaussianBlur(frame, (5, 5), 0)        # Gaussian Blur

                # Augments
                if self.subset == "training":
                    # Randomly flip the image (Over Vertical)
                    if flip:
                        frame = cv2.flip(frame, 1)

                    # Random Brightness / Contrast
                    if brightness:
                        frame = cv2.convertScaleAbs(frame, alpha=alpha_contrast, beta=beta_brightness)

                    # Random Panning
                    if panning:
                        M = np.float32([[1, 0, tx  * frame.shape[1]], [0, 1, ty  * frame.shape[0]]])
                        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

                    # Random Rotation & Zoom
                    if rotation:
                        M = cv2.getRotationMatrix2D((frame.shape[1] / 2, frame.shape[0] / 2), angle, scale)
                        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)      # Convert to YUV color space
                frame = frame / 255.0                               # Normalize pixel values to [0, 1]

                processed_frames.append(frame)

            windows.append(processed_frames)
            steering_angles.append(steering)

        return np.array(windows), np.array(steering_angles)


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