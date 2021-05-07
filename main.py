"""
We are going to use a data set from UCI Machine Learning Repository https://archive.ics.uci.edu/ml/index.php
1.  We'll be using the breast cancer data set :
    https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
2.  Click on the Data Folder
    https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
"""

import csv
import math
import operator
import os
import random

import pandas as pd


def remove_useless_data(filename: str, out_put_file_name: str):
    """
    This function is used to remove useless columns from our data set in order for our algorithm to skip processing them


    :param filename: Absolute path of the .csv file to be used
    :param out_put_file_name: Training and testing data set split ratio
    :return:
    """
    try:
        df = pd.read_csv(filename)

        # We use -99999 because most algorithms recognize this as an outlier. As opposed to dumping data
        # If you're to drop all missing data you might end up sacrificing almost 50% of your data
        df.replace('?', -99999, inplace=True)

        # If you know the name of the column skip this
        first_column = df.columns[0]
        # Delete first
        df = df.drop([first_column], axis=1)

        remove_file(out_put_file_name)

        df.to_csv(out_put_file_name, index=False)

    except Exception as ex:
        print(f'Some thing has gone wrong : Error : {ex}')
        raise ex


def remove_file(file_name):
    try:
        os.remove(file_name)
    except Exception as ex:
        print(f'Error removing file : {ex}')
        pass


def load_data_set(filename: str, split: float, data_length: int):
    """
    This function is used to load a .csv file and split it into a training data set and a test data set.
    The function takes in a filename, training and test data split ratio, an empty list of
    training data set (used to hold our training data) and an empty list of
    test data set (used to hold our test data set).


    :param filename: Absolute path of the .csv file to be used
    :param split: Training and testing data set split ratio
    :param data_length: Length of data features to consider
    :return: tuple: Returns a tuple containing our training data and test data.
    """
    try:
        training_data_set = []  # List used to hold our training data
        test_data_set = []  # List used to hold our testing data

        with open(filename, 'r') as csv_file:
            lines = csv.reader(csv_file)
            data_set = list(lines)
            for x in range(len(data_set) - 1):
                for y in range(data_length):
                    data_set[x][y] = data_set[x][y]
                if random.random() < split:
                    training_data_set.append(data_set[x])
                else:
                    test_data_set.append(data_set[x])
        return training_data_set, test_data_set
    except Exception as ex:
        print(f'Some thing has gone wrong : Error : {ex}')
        raise ex


def euclidean_distance(instance1: list, instance2: list, data_length: int):
    """
    This function is used to calculate the euclidean distance between two points. It returns a floating point number
    representing the distance between the two points i.e given two points (x1,y1) & (x2,y2)
    Euclidean distance = sqrt(sqrt(x1-x2) + sqrt(y1-y2))

    :param instance1: Initial data set to be used when calculating the distance i.e (x1, y1)
    :param instance2: Second data set to be used when calculating the distance i.e (x2, y2)
    :param data_length: The first elements of the data set to be considered
           i.e given data_length=3 with a data set [4,4,4,3,p] : the first 3 elements will be considered when
           calculating the euclidean distance
    :return: float: Returns a floating point number
    """
    try:
        distance = 0
        for x in range(data_length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)
    except Exception as ex:
        # print(f'Type error occurred : {ex}')
        return -99999


def get_nearest_neighbors(training_data_set: list, test_instance: list, k_neighbor: int):
    """
    This function is used to get the k nearest neighbors from the provided data set. This returns k similar neighbors
    from the provided training data set in the given test data set.

    :param training_data_set:
    :param test_instance:
    :param k_neighbor:
    :return:
    """
    distances = []
    length = len(test_instance) - 1
    # Get the euclidean distance of each data point in the training data set and append it to the list
    for x in range(len(training_data_set)):
        # print(f"{test_instance} : {training_data_set[x]}")
        dist = euclidean_distance(test_instance, training_data_set[x], length)
        data_instance = (training_data_set[x], dist)
        distances.append(data_instance)

    # print(f'Dist : {distances}')
    # Sort the data points using the second item in the tuple of (data set, distance)
    distances.sort(key=operator.itemgetter(1))

    neighbors = []
    for x in range(k_neighbor):
        neighbors.append(distances[x][0])
    return neighbors


def predict_response(neighbors: list):
    """
    This function is used to predict the response based on the neighbors. We allow each neighbor to vote
    for each class tribute and take the majority vote as the prediction.



    :param neighbors:
    :return:
    """
    class_votes = {}  # Dictionary to hold votes for each neighbor

    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1

    # print(f'class_votes : {class_votes}')
    # Sort the dictionary data sets in descending order i.e starting with largest at the top
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    # print(f'sorted_votes : {sorted_votes}')
    return sorted_votes[0][0]


def get_accuracy(test_data_set: list, predictions: list):
    """
    This function is used to test the accuracy of the prediction. It's the percentage of the ratio of predictions made
    to test data set multiplied by 100%.

    :param test_data_set: Test data set.
    :param predictions: Prediction where our sample data is assumed to belong to
    :return: float: Percentage accuracy of the prediction.
    """

    correct = 0
    for x in range(len(test_data_set) - 1):
        if test_data_set[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(test_data_set))) * 100.0


def main(filename, k_neighbors, new_data_samples):
    # prepare data
    split = 0.86
    # 100,000 100,000
    out_put_file_name = 'output_data.csv'
    remove_useless_data(filename, out_put_file_name)

    _training_data, _test_data = load_data_set(out_put_file_name, split, 4)

    # Generate predictions
    _predictions = []
    for x in range(len(_test_data)):
        # print(f"_test_data[x] : {_test_data[x]}")
        neighbors = get_nearest_neighbors(_training_data, _test_data[x], k_neighbors)
        result = predict_response(neighbors)
        _predictions.append(result)
        #print(f"> Predicted = {result} , actual = {_test_data[x][-1]}")

    print(f'Predictions : {_predictions}')
    # for x in range(len(_predictions)):
    _accuracy = get_accuracy(_test_data, _predictions)
    print(f'Accuracy : {_accuracy}%')


# Invoke the main function
if __name__ == '__main__':
    # 2,4
    new_data_samples = [[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 2, 2, 1, 2, 3, 2, 1], [4, 2, 2, 8, 1, 2, 3, 2, 1]]
    main(filename='dataset.data', k_neighbors=4, new_data_samples=new_data_samples)

# traningset,testset =load_data_set(filename='dataset.data', split=0.66)
# print(f'Train : {len(traningset)}')
# print(f'Test  : {len(testset)}')
#
# """ Test the euclidean distance """
# instance1 = [2, 2, 2, 'a']
# instance2 = [4, 4, 4, 'b']
# euc_distance = euclidean_distance(instance1, instance2, 3)
# print(f'Distance : {euc_distance}')
#
# """ Test the nearest neighbors """
# training_data = [[2, 2, 2, 'a'], [1, 3, 4, 'a'], [3, 2, 2, 'a'], [3, 3, 3, 'b']]
# test_instance = [4, 4, 4, 'b']
# neighbor = get_classifier(training_data, test_instance, 3)
# print(f'Neighbors : {neighbor}')
#
# predict = predict_response(neighbors=neighbor)
# print(f'Predition : {predict}')
#
# test_data = [[2, 2, 2, 'a'], [1, 3, 4, 'a'], [3, 2, 2, 'a'], [3, 3, 3, 'b']]
# predictions = ['a', 'a', 'a', 'b']
# accuracy = get_accuracy(test_data, predictions)
# print(f'Accuracy : {accuracy}')
