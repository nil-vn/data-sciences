# Credit Card Fraud Detection using Autoencoders in H2O

Frauds in the finance field are very rare to be identified. Because of that, it can do a severe damage to the financial field. It is estimated that fraud costs at least $80 billion a year across all lines of insurance. If there is a small possibility of detecting fraudulent activities, that can do a major impact on annual losses. That is why financial companies invest in machine learning as a preemptive approach to tackling fraud.
The benefits of using a machine learning approach are that,
- It helps to find hidden and implicit correlations in data.
- Faster data processing and less manual work
- Automatic detection of possible fraud scenarios.
The best way to detect frauds is anomaly detection.
# Anomaly Detection

Anomaly detection is a technique to identify unusual patterns that do not conform to the expected behaviors, called outliers. It has many applications in business from fraud detection in credit card transactions to fault detection in operating environments. Machine learning approaches for Anomaly detection;

- K-Nearest Neighbor
- Autoencoders — Deep neural network
- K-means
- Support Vector Machine
- Naive Bayes

Today we will be using Autoencoders to train the model.

## Autoencoders
Most of us are not familiar with this model. Autoencoders is an unsupervised Neural Network. It is a data compression algorithm which takes the input and going through a compressed representation and gives the reconstructed output.

Figure 1: Neural network representation of Autoencoders
Dataset
As for the dataset we will be using Credit Card Transaction dataset provided by Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud
The dataset includes 284,807 transactions. among them, 492 transactions are labeled as frauds. Because of this, the dataset is highly imbalanced. It contains only numerical variables. Feature ‘Time’ contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature ‘Amount’ is the transaction Amount, this feature can be used for example-dependent cost-sensitive learning. Feature ‘Class’ is the response variable and it takes value 1 in case of fraud and 0 otherwise.
You can find my Kaggle Kernel here: https://www.kaggle.com/maneesha96/credit-card-fraud-detection-using-autoencoders
Full code: https://github.com/Mash96/Credit-Card-Fraud-Detection
Then Let's get started!!!
Setup
We will be using H2O as the ML platform today. You can find more info here: https://www.h2o.ai
```python
import h2o
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
```
Initialize H2O server
```python
h2o.init(max_mem_size = 2) # initializing h2o server
h2o.remove_all()
```
Loading dataset using pandas data frame
```python
creditData = pd.read_csv(r"File_Path\creditcard.csv") 
creditData.describe()

# H2O method
# creditData_df = h2o.import_file(r"File_Path\creditcard.csv") 
```
Exploration
```python
creditData.shape
```
> (284807, 31)
```python
# Checking for null values in the dataset
creditData.isnull().values.any() # pandas method
creditData_h2o.na_omit() # h2o method
creditData_h2o.nacnt() # no missing values found
> False
```
In order to proceed we need to convert the pandas data frame to H2O data frame. Turns python pandas frame into an H2OFrame
```python
creditData_h2o  = h2o.H2OFrame(creditData)
# Let’s plot the Transaction class against the Frequency
labels = [‘normal’,’fraud’]
classes = pd.value_counts(creditData[‘Class’], sort = True)
classes.plot(kind = ‘bar’, rot=0)
plt.title(“Transaction class distribution”)
plt.xticks(range(2), labels)
plt.xlabel(“Class”)
plt.ylabel(“Frequency”)
```
![Figure 2](https://miro.medium.com/max/411/1*wIR9F--T1kQs7aWTqSJPog.png)
```python
fraud = creditData[creditData.Class == 1]
normal = creditData[creditData.Class == 0]
# Amount vs Class
f, (ax1, ax2) = plt.subplots(2,1,sharex=True)
f.suptitle('Amount per transaction by class')
ax1.hist(fraud.Amount, bins = 50)
ax1.set_title('Fraud List')
ax2.hist(normal.Amount, bins = 50)
ax2.set_title('Normal')
plt.xlabel('Amount')
plt.ylabel('Number of Transactions')
plt.xlim((0, 10000))
plt.yscale('log')
plt.show()
```
![Figure 3](https://miro.medium.com/max/404/1*a8yDQUpb7MWUW0y_GJDD8g.png)
```python
# time vs Amount
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud List')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in seconds)')
plt.ylabel('Amount')
plt.show()
```
![Figure 4](https://miro.medium.com/max/406/1*Iy6VMcvz2WSn8btOaY_EkA.png)
```python
#plotting the dataset considering the class
color = {1:'red', 0:'yellow'}
fraudlist = creditData[creditData.Class == 1]
normal = creditData[creditData.Class == 0]
fig,axes = plt.subplots(1,2)
axes[0].scatter(list(range(1,fraudlist.shape[0] + 1)), fraudlist.Amount,color='red')
axes[1].scatter(list(range(1, normal.shape[0] + 1)), normal.Amount,color='yellow')
plt.show()
```
![Figure 5: Frauds vs Normals](https://miro.medium.com/max/384/1*MRmiK5-LXpfbc4r4SBGccg.png)
## Preparing Data
The Time variable is not giving an impact on the model prediction. This can figure out from data visualization. Before moving on to the training part, we need to figure out which variables are important and which are not. So we can drop the unwanted variables.
```python
features= creditData_h2o.drop(['Time'], axis=1)
```
Split the data frame as training set and testing set keeping 80% for the training set and rest to the testing set.
```python
train, test = features.split_frame([0.8])
print(train.shape)
print(test.shape)
> (227722, 30)
> (57085, 30)
```
Our dataset has a lot of non-fraud transactions. Because of this for the model training, we only send non-fraud transactions. So that the model will learn the pattern of normal transactions.
```python
# converting to pandas dataframe
train_df = train.as_data_frame()
test_df = test.as_data_frame()
train_df = train_df[train_df['Class'] == 0]
# drop the Class variable
train_df = train_df.drop(['Class'], axis=1)
Y_test_df = test_df['Class'] # true labels of the testing set
test_df = test_df.drop(['Class'], axis=1)
train_df.shape
> (227335, 29)
```
## Model Building
```python
train_h2o = h2o.H2OFrame(train_df) # converting to h2o frame
test_h2o = h2o.H2OFrame(test_df)
x = train_h2o.columns
```
When building the model, 4 fully connected hidden layers were chosen with, [14,7,7,14] number of nodes for each layer. First two for the encoder and last two for the decoder.
```python
anomaly_model = H2ODeepLearningEstimator(activation = "Tanh",
                               hidden = [14,7,7,14],
                               epochs = 100,
                               standardize = True,
                                stopping_metric = 'MSE', 
                                loss = 'automatic',
                                train_samples_per_iteration = 32,
                                shuffle_training_data = True,     
                               autoencoder = True,
                               l1 = 10e-5)
anomaly_model.train(x=x, training_frame = train_h2o)
```
## Model Evaluation
Variable Importance : In H2O there is a special way of analyzing which variables are giving higher impact on the model.
anomaly_model._model_json['output']['variable_importances'].as_data_frame()

## Visualization
```python
# plotting the variable importance
rcParams['figure.figsize'] = 14, 8
#plt.rcdefaults()
fig, ax = plt.subplots()

variables = anomaly_model._model_json['output']['variable_importances']['variable']
var = variables[0:15]
y_pos = np.arange(len(var))

scaled_importance = anomaly_model._model_json['output']['variable_importances']['scaled_importance']
sc = scaled_importance[0:15]

ax.barh(y_pos, sc, align='center', color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(variables)
ax.invert_yaxis()
ax.set_xlabel('Scaled Importance')
ax.set_title('Variable Importance')
plt.show()
```
![Figure 6](https://miro.medium.com/max/845/1*Lms7bYGKkpSmDwoVbtEqrA.png)
```python
# plotting the loss
scoring_history = anomaly_model.score_history()
%matplotlib inline
rcParams['figure.figsize'] = 14, 8
plt.plot(scoring_history['training_mse'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
```
![Figure 7](https://miro.medium.com/max/854/1*g5mH7ilWkNjBaGXqEM8fKA.png)
The testing set has both normal and fraud transactions in it. The Autoencoder will learn to identify the pattern of the input data. If an anomalous test point does not match the learned pattern, the autoencoder will likely have a high error rate in reconstructing this data, indicating anomalous data. So that we can identify the anomalies of the data. To calculate the error, it uses Mean Squared Error(MSE)
```python
test_rec_error = anomaly_model.anomaly(test_h2o) 
# anomaly is a H2O function which calculates the error for the dataset
# converting to pandas dataframe
test_rec_error_df = test_rec_error.as_data_frame()
# plotting the testing dataset against the error
test_rec_error_df['id']=test_rec_error_df.index
rcParams['figure.figsize'] = 14, 8
test_rec_error_df.plot(kind="scatter", x='id', y="Reconstruction.MSE")
plt.show()
```
![Figure 8: Anomalies in testing set](https://miro.medium.com/max/842/1*xGCPPGGtEWdg7RQfmBOtuQ.png)
```python
# predicting the class for the testing dataset
predictions = anomaly_model.predict(test_h2o)
error_df = pd.DataFrame({'reconstruction_error': test_rec_error_df['Reconstruction.MSE'],
                        'true_class': Y_test_df})
error_df.describe()
```
![Figure 9](https://miro.medium.com/max/507/1*Eg5Q8HXM6HqEWhGW4Oissg.jpeg)
```python
# reconstruction error for the normal transactions in the testing dataset
fig = plt.figure()
ax = fig.add_subplot(111)
rcParams['figure.figsize'] = 14, 8
normal_error_df = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < 10)]
_ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)
```
![Figure 10](https://miro.medium.com/max/837/1*nDQ8PxR-H8YDUcu4pjb5dQ.png)
```python
# reconstruction error for the fraud transactions in the testing dataset
fig = plt.figure()
ax = fig.add_subplot(111)
rcParams['figure.figsize'] = 14, 8
fraud_error_df = error_df[error_df['true_class'] == 1]
_ = ax.hist(fraud_error_df.reconstruction_error.values, bins=10)
```
![Figure 11](https://miro.medium.com/max/818/1*BxjhMKoFHCTr6tVm3JJSaA.png)

## ROC Curve
```python
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();
```
![Figure 12](https://miro.medium.com/max/840/1*xmiiEPH_k-w0ZF17iaS78w.png)

The accuracy is 0.9718
## Precision & Recall

Since the data is highly imbalanced, it cannot be measured only by using accuracy. Precision vs Recall was chosen as the matrix for the classification task.
Precision: Measuring the relevancy of obtained results.
[ True positives / (True positives + False positives)]
Recall: Measuring how many relevant results are returned.
[ True positives / (True positives + False negatives)]
True Positives — Number of actual frauds predicted as frauds
False Positives — Number of non-frauds predicted as frauds
False Negatives — Number of frauds predicted as non-frauds.
```python
precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
plt.plot(recall, precision, 'b', label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
```
![Figure 13](https://miro.medium.com/max/835/1*y2aychfyiaB4euYh5iWcbA.png)

We need to find a better threshold that can separate the anomalies from normal. This can be done by calculating the intersection of the Precision/Recall vs Threshold graph.
```python
plt.plot(th, precision[1:], label="Precision",linewidth=5)
plt.plot(th, recall[1:], label="Recall",linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()
```
![Figure 14](https://miro.medium.com/max/835/1*1m4vg1LefOe1P4g3ZWROoQ.png)
```python
# plot the testing set with the threshold
threshold = 0.01
groups = error_df.groupby('true_class')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Fraud" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();
```
![Figure 15](https://miro.medium.com/max/842/1*QYv2hGzaU9X6FwJc-cwt0w.png)

## Confusion Matrix
```python
import seaborn as sns
LABELS = ['Normal', 'Fraud']
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
```
![Figure 16](https://miro.medium.com/max/685/1*7UOWqffPGmdzfN_H3x-M2Q.png)

## Classification Report
```python
csr = classification_report(error_df.true_class, y_pred)
print(csr)
```
![Figure 17](https://miro.medium.com/max/713/1*i9NL9YhwON_87p5dYbSd0w.jpeg)

## Conclusion
Our model is catching most of the fraudulent data. In Autoencoders, it gives a good accuracy. But if we look into Precision and Recall of the dataset, it is not performing enough. As I mentioned earlier, there are other anomaly detection methods that perform well in highly imbalanced datasets.
I have tried more methods on this dataset. So I will see you soon with those. :)

## References
https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd
