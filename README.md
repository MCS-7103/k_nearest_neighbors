**k-nearest neighbors algorithm**

* k-nearest neighbors algorithm (k-NN or KNN) is a non-parametric classification method used for classification and regression. The input consists of the k closest training examples in data set.
*  In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
* In k-NN regression, the output is the property value for the object. This value is the average of the values of k nearest neighbors.
* Since this algorithm relies on distance for classification, if the features represent different physical units or come in vastly different scales then normalizing the training data can improve its accuracy
* a useful technique can be to assign weights to the contributions of the neighbors, so that the nearer neighbors contribute more to the average than the more distant ones. For example, a common weighting scheme consists in giving each neighbor a weight of 1/d, where d is the distance to the neighbor.

**Distance between points - Euclidean distance**
The distance between any two points on the real line is the absolute value of the numerical difference of their coordinates.Thus if p  and q are two points on the real line, then the distance between them is given by:

![](https://i.imgur.com/B03FPrX.png)

In the Euclidean plane, let point p have Cartesian coordinates ( p 1 , p 2 ) and let point q have coordinates ( q 1 , q 2 ). Then the distance between p and q is given by:

![](https://i.imgur.com/MrtRbnM.png)

In general, for points given by Cartesian coordinates in n-dimensional Euclidean space, the distance is:

![](https://i.imgur.com/4M3JRGP.png)

**CHOOSING THE VALUE OF K**
* If you randomly select the K value and get into this situation, this means that your K value is not optimized so you need to optimize it based on your dataset. How do you choose the optimal value of K?
* There are no pre-defined statistical methods to find the most favorable value of K.
* Initialize a random K value and start computing.
* Choosing a small value of K leads to unstable decision boundaries.
* The substantial K value is better for classification as it leads to smoothening the decision boundaries.
* Derive a plot between error rate and K denoting values in a defined range. Then choose the K value as having a minimum error rate.

**KNN overfitting and underfitting**
* The value of k in the KNN algorithm is related to the error rate of the model. 
* A small value of k could lead to overfitting as well as a big value of k can lead to underfitting. 
* Overfitting imply that the model is well on the training data but has poor performance when new data is coming. 
* Underfitting refers to a model that is not good on the training data and also cannot be generalized to predict new data.
* To avoid overfitting,
* To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test. 

**Industrial use cases of the KNN Algorithm**
* Recommender systems
* Suggest you which item you are likely to buy based on the items in your shopping cart, or the item you’re currently viewing.
* Concept search
* Extra concepts from a set of documents available in data sets on the internet

**KNN Implementation using Python**
We follow the steps below;
* Handle data: open the dataset from csv & split it into training and test data	
* Calculate the Euclidean distance between the data sets
* Locate K most similar/nearest data instances	
* Generate a response from a set of data instances and determine where it belong	
* Summarize the accuracy of the prediction
* Combine the functions into one executable function
