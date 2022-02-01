# DISTANCE BASED CLASSIFICATION-IRIS DATASET

### **IRIS DATASET**

The Iris Flowers Dataset involves predicting the flower species given measurements of iris flowers.

It is a multi-class classification problem. The number of observations for each class is balanced. 

There are 150 observations with 4 input variables and 1 output variable. The variable names are as follows:

1. Sepal length in cm.

2. Sepal width in cm.

3. Petal length in cm.

4. Petal width in cm.

5. Class (Iris Setosa, Iris Versicolor, and Iris Virginica).

The baseline performance of predicting the most prevalent class is a classification accuracy of approximately 26%.

##### Loading Dataset

```python
df = pd.read_csv("iris.data")
```

To load the dataset from the file, read_csv() of pandas is used. 

## <u>IMPLEMENTATION</u>

### MODULES USED

**Numpy** - NumPy is a Python library used for working with arrays.

**Pandas** - It provides ready to use high-performance data structures and data analysis tools. Pandas module runs on top of NumPy and it is popularly used for data science and data analytics.

**Statistics** - This module provides functions for calculating mathematical statistics of numeric data. 

**Matplotlib** - Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

### **DATA PREPROCESSING**

Data preprocessing is a process of preparing the raw data and making it suitable for a machine learning model. Data preprocessing increases the accuracy and efficiency of a machine learning model. 

STEPS:

1. Finding missing data

2. Encoding Categorical Data

##### **Ways to handle missing data:**

·     **By deleting the particular row**

Common way of dealing with NULL values is to delete that specific row (or column). This method is not an efficient method as removing data lead to loss of information.

·     **By calculating the mean**

In this method, we calculate the mean of that column (or row) which contains the missing value, and fill it with calculated mean value. This strategy is useful for the features which have numeric data.

```python
print(df.isnull().sum())
```

NOTE:-

Iris dataset has no missing value.

##### **Categorical Data:**

Two kinds of categorical data:-

·     **Ordinal Data -** The categories have an inherent order. In Ordinal data, while encoding, one should retain the information regarding the order in which the category is provided. Example: highest degree possessed by a person.

·     **Nominal Data -** The categories do not have an inherent order. While encoding Nominal data, we have to consider the presence or absence of a feature. In such a case, no notion of order is present. Example: the city a person lives in.

NOTE:-

In iris dataset, there is no features which have categorical data. Class label has to be encoded. 

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
# Iris-setosa 0
# Iris-versicolor 1
# Iris-virginica 2
```

### EXPLORATORY DATA ANALYSIS

Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.

Scatterplots based on 2 feature at a time are plotted to understand the pattern of data. 

Based on the scatterplots, it is concluded that, we can make good prediction even with only two features. So, the features petal-length and petal-width are used for classification in this project.

### TRAIN AND TEST DATA

60% of the dataset is used for training 40% of it is used as 40%.

<u>Training Data</u>

60% of 150 = 90

First 30 of each class is taken for training. 

Indices [0:30], [50:80], [100:130]

<u>Testing Data</u>

40% of 150 = 60

Last 20 of each class is taken for testing.

Indices [30:50], [80:100], [130:150]

### TRAINING PHASE

Training phase includes finding the centroid (mean) of each class. Since we have 3 classes and 4 features, we have a 3 centroid each containing 4 values. Mean of each class is calculated separate. 

<u>Calculating Mean:</u>

Mean is calculated by adding the train-data column-wise and dividing my the length of train-data (here 30). This function returns 1x4 array.

```python
def mean_0(X):
    # Calculates mean of each column
    m,n = X.shape
    sums = 0
    for row in X:
        sums += row
    return sums/m
```

The centroids are plotted along with the train-data.

### DISTANCE METRICS

Functions are written for calculating Euclidean distance, City Block/Manhattan distance, Chess Board distance, Mahalanobis distance, Correlation distance, Cosine distance, Bray-Curtis distance, Canberra distance and Minkowski distance. 

Each of these functions takes two points (test point and centroid) as parameters and calculates the distance based on its formula. Since, we are considering only two features, distance between two points in 2D is calculated. 

Mahalanobis distance calculation makes use of inverse of covariance of the train_data. 

##### Euclidean Distance

$$
d(x,y) = \sqrt{(x_1-y_1)^2+(x_2-y_2)^2+...+(x_n-y_n)^2}
$$

```python
def euclidean_distance(p1,p2):
    distance = pow(sum([(a - b) ** 2 for a, b in zip(p1, p2)]),0.5)
    return distance
```

##### City Block/Manhattan

$$
d(x,y)= |x_1 - x_2| + |y_1 - y_2| + ... + |x_n - y_n|
$$



```python
def manhattan_distance(p1,p2):
    distance = 0
    for i in range(len(p1)):
        distance += abs(p1[i] - p2[i])
    return distance
```

##### Chess Board Distane

$$
d(x,y) = max(|x_1 - x_2|,|y_1 - y_2|)
$$

```python
def chessboard_distance(p1,p2):
    distance = abs(p1[0] - p2[0])
    for i in range(1,len(p1)):
        distance = max(distance,abs(p1[i] - p2[i]))
    return distance
```

##### Mahalanobis Distance

$$
d(x,y)=(x-y)^T.C.(x-y)
$$

```python
def mahalanobis_distance(p1,p2,X): #p1 is model, p2 is the test point
    # X is inverse cov matrix
    distance = np.dot(np.dot(np.subtract(p2,p1).T,np.array(X)),np.subtract(p2,p1))
    return distance
```

##### Correlation Distance

$$
d(x,y) = 1 - (x - Mean[x]).(y - Mean[y])/(Norm(x - Mean[x])Norm(y - Mean[]y)) 
$$

```python
def correlation_distance(p1,p2):
    norm_p1 = 0
    norm_p2 = 0
    for i in range(len(p1)):
        norm_p1 += (p1[i] - st.mean(p1))**2
        norm_p2 += (p2[i] - st.mean(p2))**2
    norm_p1 = norm_p1**0.5
    norm_p2 = norm_p2**0.5
    s = 0
    for i in range(len(p1)):
        s += (p1[i] - st.mean(p1))*(p2[i] - st.mean(p2))
    distance = 1 - s/(norm_p1*norm_p2)
    return distance
```

##### Cosine Distance

$$
\text{cosine distance} = 1 - \text{cosine similarity}(A,B)
$$

$$
\text{cosine similarity} = \cos(\theta) = {\mathbf{A} \cdot \mathbf{B} \over \|\mathbf{A}\| \|\mathbf{B}\|} =\large \frac{ x_1 * x_2 + y_1 * y_2  }{ \sqrt{x_1^2+y_1^2 }  \sqrt{x_2^2 + y_2^2}}
$$

```python
def cosine_distance(p1,p2):
    norm_p1 = 0
    norm_p2 = 0
    for i in range(len(p1)):
        norm_p1 += p1[i]**2
        norm_p2 += p2[i]**2
    norm_p1 = norm_p1**0.5
    norm_p2 = norm_p2**0.5
    s = 0
    for i in range(len(p1)):
        s += p1[i]*p2[i]
    distance = 1 - s/(norm_p1*norm_p2)
    return distance
```

##### Bray-Curtis Distance

$$
\large BC_d=\sum_{i=0}^{n} \frac{|x_i-x_j|}{(x_i-x_j)}
$$

```python
def bray_curtis_distance(p1,p2):
    s1 = 0
    s2 = 0
    for i in range(len(p1)):
        s1 += abs(p1[i] - p2[i])
        s2 += abs(p1[i] + p2[i])
    distance = s1/s2
    return distance
```

##### Canberra Distance

$$
\large d(p,q) = \sum_{i=0}^{n} \frac{|p_i-q_i|}{|p_i|+|q_i|}
$$

```python
def canberra_distance(p1,p2):
    distance = 0
    for i in range(len(p1)):
        s1 = abs(p1[i] - p2[i])
        s2 = abs(p1[i] + p2[i])
        distance += s1/s2
    return distance
```

##### Minkowski Distance

$$
\large d(x,y) = (|x_1 - y_1|^p + |x_2 - y_2|^p)^{1/p}
$$

```python
def minkowski_distance(p1,p2,p):
    s = 0
    for i in range(len(p1)):
        s += abs(p1[i] - p2[i])**p
    distance = s**(1/p)
    return distance
```

### MISCLASSIFICATION ERROR RATE

```python
def MER_Error(X,Y):
    correct_count = 0
    for i in range(len(X)):
        if(X[i] == Y[i]):
            correct_count = correct_count + 1
    MER_val = 1 - (correct_count/len(X))
    return MER_val
```

This function make count of correct classification by checking if the predicted label is same as the original label. Each time, when they are same, correct_count is incremented by one. Number of misclassification is calculated by no. of predicted points minus correctly predicted. Misclassification error rate is no. of wrong predictions by total no. of predicted points.

### TESTING PHASE

In testing phase, for each point, distance from the point to all 3 centroids are calculated. Minimum of the 3 distance is taken for classification. The point is classified to the class whose centroid is the nearest (distance is minimum). The same process is repeated for each of the distance metrics. After classification, the predicted class label is compared with the actual class label. If both are different, its a misclassification.

##### Euclidean Distance Analysis

The test data is classified using euclidean distance and the results are analyzed.

```python
# minimum distance
min_dist = MAX
# predicted labels
predicted = [0]*len(X_test)
#Actual label -> Y_test
for i in range (len(X_test)):
    for j in range(0,3):
        distance = euclidean_distance([model[j,2],model[j,3]],[X_test[i,2],X_test[i,3]])        
        if(distance < min_dist):
            min_dist = distance
            lbl = j;
    predicted[i] = lbl;
    #reset min_dist
    min_dist = MAX  
```

The predicted label and actual label are printed so as to compare both.

The MER of Euclidean Distance Model was estimated to be 0.0333.

##### Analysis using a Sample Test Point (Euclidean Distance Model)

A random sample point for selected from the test data and analysis is done. The actual and predicted label is printed.

```python
import random
# Select a random test plot
pt = random.randint(0,len(Y_test))
print("Sample Test Point Id",pt)
print("True Class of Sample Test Point -> ",Y_test[pt])
print("Predicted Class of Sample Test Point -> ",predicted[pt])
```

The distance from 3 centroids to the selected point is plotted.

*<u>**NOTE**</u>*:

This step was repeated several times so as to observe a case of misclassification. In the plot of distance from centroid to the point, it was clear that the distance from point to the centroid of actual class and centroid of predicted class were same. The misclassification was due to the equal distance. In this case, the point was classified to the class with which the distance is calculated first.

### MER AND ACCURACY

MER is calculated for all the distance classification model, and its corresponding accuracy is calculated and then, stored in an array.

MER of different distance model is plotted in the form of a bar chart. 

Similarly, the accuracy of each model is also plotted.