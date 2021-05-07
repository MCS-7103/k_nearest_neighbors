import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def load_data_set(filename: str, test_size: float):
    """
    Load the .csv file and split arrays or matrices into random train and test subsets \
    while holding out test_size i.e 40% of the data for testing (evaluating) our classifier:\

    :param filename: Absolute path to the dataset
    :param test_size: Percentage of data to be helf for testing (evaluating) our classifier
    :return:
    """
    # Watch out for useless data i.e the id column
    df = pd.read_csv(filename)

    # We use -99999 because most algorithms recognize this as an outlier. As opposed to dumping data
    # If you're to drop all missing data you might end up sacrificing almost 50% of your data
    df.replace('?', -99999, inplace=True)

    # Drop the Id column since it's useless and can made our algorithm output inaccurate data
    df.drop(['id'], 1, inplace=True)

    # For the features, represents everything except the classification column
    x = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])  # For the labels

    # scikit-learn randomly splits data into training and test sets using train_test_split function
    return train_test_split(x, y, test_size=test_size)


def get_classifier(x_train: list, y_train: list, neighbors: int):
    """
    Fit the k-nearest neighbors classifier from the training dataset.

    :param x_train: {array-like, sparse matrix} of shape (n_samples, n_features) or \
                    (n_samples, n_samples) if metric='precomputed' Training data.
    :param y_train: y : {array-like, sparse matrix} of shape (n_samples,) or \
                    (n_samples, n_outputs) Target values.
    :param neighbors: Number of K neighbors to consider
    :return: self : KNeighborsClassifier , The fitted k-nearest neighbors classifier.
    """
    classifier = KNeighborsClassifier(n_neighbors=neighbors)
    classifier.fit(x_train, y_train)
    return classifier


def get_accuracy(x_test: list, y_test: list, classifier: KNeighborsClassifier):
    """
    Return the mean accuracy on the given test data and labels. \
    In multi-label classification, this is the subset accuracy \
    which is a harsh metric since you require for each sample that \
    each label set be correctly predicted. \

    :param x_test: array-like of shape (n_samples, n_features) Test samples.
    :param y_test: array-like of shape (n_samples,) or (n_samples, n_outputs) True labels for `X`.
    :param classifier:
    :return: score : float Mean accuracy of ``self.predict(X)`` wrt. `y`.
    """
    return classifier.score(x_test, y_test)


def get_predictions(classifier: KNeighborsClassifier, data_samples: list):
    """
    Predict the class labels for the provided data.


    :param classifier:
    :param data_samples: array-like of shape (n_queries, n_features), or (n_queries, n_indexed)
                        if metric == 'precomputed' Test samples.
    :return:  ndarray of shape (n_queries,) or (n_queries, n_outputs)
                Class labels for each data sample.
    """
    numpy_array = np.array(data_samples)
    numpy_array = numpy_array.reshape(len(data_samples), -1)
    return classifier.predict(numpy_array)


def main(filename: str, test_size: float, data_samples: list, neighbors: int):
    x_train, x_test, y_train, y_test = load_data_set(filename, test_size)
    print(f"x_train : {len(x_train)} : {len(x_test)}")
    print(f"y_train : {len(y_train)} : {len(y_test)}")

    classifier = get_classifier(x_train, y_train, neighbors)

    prediction = get_predictions(classifier, data_samples)
    print(f'prediction : {prediction}')

    accuracy = get_accuracy(x_test, y_test, classifier)
    print(f'accuracy : {accuracy}')


# Invoke the main function
if __name__ == '__main__':
    csv_filename = 'breast-cancer-wisconsin.data'
    test_percentage = 0.2
    new_samples = [[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 2, 2, 1, 2, 3, 2, 1], [4, 2, 2, 8, 1, 2, 3, 2, 1]]
    k_neighbors = 4
    print(f"new_samples : {new_samples}")
    main(csv_filename, test_percentage, new_samples, k_neighbors)
