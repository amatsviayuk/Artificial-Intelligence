import cv2
from pathlib import Path
import math
import operator

training_feature_vector = []  # Training feature vector that stores features to detect color


# Calculation of Euclidean distance
def calculate_euclidean_distance(variable1, variable2, length):
    distance = 0
    for x in range(length):
        distance += (variable1[x] - variable2[x]) ** 2
    return math.sqrt(distance)


# Get k nearest neighbors
def k_nearest_neighbors(test_instance, k):
    distances = []
    length = len(test_instance)
    for x in range(len(training_feature_vector)):  # distances between the test instance and training feature vectors.
        dist = calculate_euclidean_distance(test_instance, training_feature_vector[x][:-1], length)
        distances.append((training_feature_vector[x], dist))
    distances.sort(key=operator.itemgetter(1))  # sorting neighbours
    neighbors = [distances[x][0] for x in range(k)]
    return neighbors


# Votes of neighbors
def response_of_neighbors(neighbors):
    all_possible_neighbors = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in all_possible_neighbors:
            all_possible_neighbors[response] += 1
        else:
            all_possible_neighbors[response] = 1
    sorted_votes = sorted(all_possible_neighbors.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


# Extract color histogram features from image to store in training feature vector
def color_histogram_of_image(image):
    chans = cv2.split(image)
    features = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return features


# Label training data based on file name
def color_histogram_of_training_image(img_path):
    img_name = img_path.name
    if "black" in img_name:
        label = "black"
    elif "blue" in img_name:
        label = "blue"
    elif "grey" in img_name:
        label = "grey"
    elif "red" in img_name:
        label = "red"
    elif "white" in img_name:
        label = "white"
    elif "green" in img_name:
        label = "green"
    else:
        raise ValueError(f"Unknown color label for file {img_name}")

    image = cv2.imread(str(img_path))
    features = color_histogram_of_image(image)
    features.append(label)
    training_feature_vector.append(features)


# Load training data
def training():
    base_path = Path(__file__).parent / "training_dataset"
    colors = ["black", "blue", "grey", "red", "white", "green"]
    for color in colors:
        color_path = base_path / color
        if not color_path.exists():
            raise FileNotFoundError(f"Directory {color_path} does not exist")
        for img_path in color_path.iterdir():
            if img_path.is_file():
                color_histogram_of_training_image(img_path)


# Main function to predict color
def main(image):
    test_feature_vector = color_histogram_of_image(image)
    k = 3  # Number of neighbors
    neighbors = k_nearest_neighbors(test_feature_vector, k)
    return response_of_neighbors(neighbors)


# Function to detect color from an image
def detect_color(box_image):
    if not training_feature_vector:
        training()
    return main(box_image)


# Example usage
if __name__ == "__main__":
    print("Current Working Directory:", Path.cwd())
    test_image_path = Path('szary.jpg')  # Replace with your test image path
    if not test_image_path.is_file():
        raise FileNotFoundError(f"Test image {test_image_path} does not exist")
    test_image = cv2.imread(str(test_image_path))
    predicted_color = detect_color(test_image)
    print(f"Detected color: {predicted_color}")
