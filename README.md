# 5790_KNN
# Overview
In this assignment, I have implemented a K-Nearest Neighbors (KNN) classifier for spam detection. The KNN algorithm is applied with various values of 'k' on datasets provided in the 'spam_train.csv' and 'spam_test.csv' files. The code includes functions for data processing, feature selection, and KNN prediction with and without z-score normalization. It evaluates the classifier's accuracy and provides visualization of the results.
# Installation
To run this code, it is recommended to use the Spyder IDE, which can be downloaded via Anaconda. Ensure that you have the following Python libraries installed:
• pandas
• numpy
• matplotlib
# Usage
## Functions
`knn_prediction_without_norm(df_name)`

This function performs KNN prediction without normalization. For each test instance, it calculates the
Euclidean distance to all training instances, sorts them, takes the nearest 'k' neighbors, and predicts
the label based on the majority class. The accuracy is calculated for a list of 'k' values.

`z_score_norm(data)`

This function performs z-score normalization on the data.

`knn_prediction_with_norm()`

This function performs KNN prediction with z-score normalization. For each test instance, it calculates the Euclidean distance to all training instances, sorts them, takes the nearest 'k' neighbors, and predicts the label based on the majority class. The accuracy is calculated for a list of 'k' values.
# Main Loop
A for-loop iterates through different 'k' values, calculating KNN predictions with and without normalization. It also generates a result table for the first 50 test instances for each 'k' value, mapping the predicted labels to 'spam' and 'no'.
# Results
For each 'k' value, the code outputs the accuracy of the KNN classifier, showing how well it performs on the test dataset. The accuracy is calculated for a range of 'k' values, helping to identify the best 'k' for the task of spam detection. Additionally, the code provides a result table that maps the predicted labels to 'spam' and 'no' for the first 50 test instances for each 'k' value, aiding in visualizing the classifier's performance.
