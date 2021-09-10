# Identifying ‘At Risk’ Cancer Patients Based OnMicroRNA Biomarkers Using an Ensemble Approach

Names: Christabelle Pabalan and Kyle Brooks

### Introduction
According to the National Cancer Institute, “[a]pproximately 39.5% of [individuals in the US] will be diagnosed with cancer at some point during their lifetimes”.  As the prevalence of cancer cases increases, there is a simultaneous urgency for cancer prognosis. Moreover, as the applications of machine learning on personalized healthcare expand, research focused on cancer prognosis and prediction have emerged. This paradigm shift in machine learning towards cancer prognosis has the potential to classify patients into high and low risk groups and prevent the initial development of cancer. Recently, there have been several reports that reveal microRNAs to be promising biomarkers for cancer. In this paper, we assess the performance of machine learning models for early prediction on an ensemble of 977 microRNAs from 21 sources. Some of the sources differ in the number of classes to predict so different classification models are trained for the corresponding number of classes. The predictions for each of these models are combined and submitted to the kaggle competition, resulting in an accuracy score of 0.73039 (on 40% of the testing data).

## Methodology
### 2.1 Dataset Description 
The dataset we are using to create our model is composed of 977 circulating microRNAs (miRNAs) that have been used as potential biomarkers for identifying different types of cancer, since they are involved in cancer development. The feature problem_id indicates which dataset the microRNA values come from (of 21 sources). The target is five values ranging from 0-5 that we are trying to classify for each observation.

### 2.2 Research Design
The central question we aim to answer is whether or not we can accurately predict cancer using machine learning techniques on microRNAs. However, there are several considerations we took into account to solidify our methodology. The first consideration is the lack of sufficient data; for each problem statement, we are presented with a wide matrix. As the number of instances decrease and model complexity increases, it becomes highly likely that the model will overfit to the nuances and noise of the training data. Consequently, the model will have poor performance on future data. We address this by taking a large ensemble of weak learners as a means of regularization. The second consideration is that our data is a composition of several classification tasks joined together. 

The problems correspond to different classification problems (e.g. some are binary, three class, etc.). Therefore, we chose to create a different model for each classification type, and combine all of our predictions. The last consideration is that for some problems, there is a significant class imbalance. The problem ID’s with the most considerable case of class imbalance is PID 4 and 18. As a result, we isolated these problems and applied synthetic minority oversampling to alleviate this issue. While there are other problems with class imbalance, we wanted to avoid separating the problems as much as possible in order to preserve more data and decrease the sensitivity to overfitting.

### 2.3 Methods and Measures 
- Synthetic Minority Oversampling (SMOTE): A sampling technique that addresses class imbalance by generating synthetic instances of the minority class using k-nearest neighbors.
- Random Forest Classifier: A collection of decision tree predictors that each contain a random subset of features. The features are split by the highest purity of nodes.
- Extra Trees Classifier: A variation of random forest that introduces more variability by creating feature splits at random rather than by highest purity.
- LazyPredict: An autoML Python package that applies all of the common machine learning algorithms to a dataset and presents common metrics based on the task. 
- Shuffle Split: An iterative cross validation technique that randomly generates a training and test set from the data during every iteration. This enhances variability, providing a holistic estimation on how the model can be expected to generally perform on unseen data.

