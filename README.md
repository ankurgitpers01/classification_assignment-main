# Machine Learning Classification Assignment

## 1. Problem Statement

Implementation of five machine learning classification models
to classify breast tumors into Benign and Malignant categories.

An interactive Streamlit application is developed to evaluate
and compare model performance.

## 2. Dataset Description

Dataset: Breast Cancer Wisconsin Diagnostic Dataset

Source:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

- Total instances: 569
- Input features: 30
- Target classes: 2
- Classification type: Binary

Target:
0 = Benign
1 = Malignant

## 3. GitHub Repository Link

https://github.com/ankurgitpers01/classification_assignment-main

## 4. Models Used

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

### Evaluation Metrics

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC

### Model Comparison Table 

ML Model Name	           Accuracy	    AUC	        Precision	    Recall	      F1	         MCC
Logistic Regression	    0.964912281	0.996031746  0.975	        0.928571429	 0.951219512	0.924518119
Decision Tree	          0.929824561	0.924603175	 0.904761905	  0.904761905	 0.904761905	0.849206349
KNN	                    0.956140351	0.982308201	 0.974358974 	0.904761905	   0.938271605	0.905823874
Naive Bayes	            0.921052632	0.989087302	 0.923076923 	0.857142857  	 0.888888889	0.82916172
Support Vector Machine	0.973684211	0.994708995	 1	          0.928571429	   0.962962963	0.944154951


### Model Observations
**Logistic Regression

Achieved an accuracy of 96.49% and an AUC of 99.60%. It demonstrated excellent classification performance with a high precision of 97.50% and recall of 92.86%. It is the second-best overall model based on F1 Score.

**Decision Tree
Achieved an accuracy of 92.98% and an F1 Score of 90.48%. It provided reasonable classification performance but performed lower than the other models on most evaluation metrics.

**KNN

Achieved an accuracy of 95.61% and an AUC of 98.23%. It showed good classification performance with 97.44% precision and 93.83% F1 Score. It performed better than Decision Tree and Naive Bayes.

**Naive Bayes

Achieved an accuracy of 92.11% and an AUC of 98.91%. Although it achieved a good AUC score, its recall of 85.71% and F1 Score of 88.89% were comparatively lower.

**Support Vector Machine

Achieved the highest accuracy of 97.37%, precision of 100%, F1 Score of 96.30% and MCC of 0.9442. It demonstrated the best overall classification performance among the five models.

**Overall Winner: Support Vector Machine (SVM)**

Support Vector Machine achieved the best overall performance among the five classification models.

It obtained the highest Accuracy of 97.37%, Precision of 100%, F1 Score of 96.30%, and MCC Score of 0.9442.

It also achieved a Recall of 92.86% and an AUC Score of 0.9947.

Although Logistic Regression achieved the highest AUC score of 0.9960, SVM performed better across most of the other evaluation metrics.

Therefore, Support Vector Machine is selected as the overall best-performing classification model for the Breast Cancer Wisconsin Diagnostic Dataset.

## 5. Streamlit Application

Features:

- CSV upload
- Model selection
- Evaluation metrics
- Confusion matrix
- Classification report
- Model comparison
- Accuracy visualization

Live App:
https://classificationassignment-main-2cwr3mda6burrixpwtuxvb.streamlit.app/

## 6. Repository Structure

- app.py
- train.py
- requirements.txt
- README.md
- test_data.csv
- model/
- screenshots/

## 7. Deployment

Deployed using Streamlit Community Cloud.

## 8. Conclusion

Five classification algorithms were implemented and evaluated
using the Breast Cancer Wisconsin Diagnostic Dataset.

An interactive Streamlit application was developed to compare
model performance using six evaluation metrics.

