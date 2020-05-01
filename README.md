# Predicting the Survival of Titanic Passengers


## Project Summary
In the Kaggle competition [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic), we are given a dataset about passengers on RMS Titanic, the ship that sank after colliding with and iceberg in 1912, resulting in the death of 1502 out of 2224 passengers and crew. Were some groups of passengers more likely to survive than others? We are asked to build a predictive model that to answer this question using passenger data.

In this exercise, I did the following:

- **EDA** (Exploratory Data Analysis): analyzed the dataset by calculating statistics and visualization (bar charts, point charts, etc.).  
- **Feature Engineering**: extracted additional features from the raw data and handled missing data in different ways.
- **Modeling**: modeled data using classic machine learning algorithms including *Logistic Regression*, *Random Forest*, *KNN* and *XGBOOST*; used *cross validation* and *grid search* methods to tune model hyperparameters; and compared performance of different models.
- **Prediction**: generated prediction on test dataset and submitted prediction on Kaggle. I achieved an accuracy score of 77.5%.


## Tools
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn

## Resources
In order to improve the performance of the machine learning model, I borrowed ideas from some articles online.

- [Kaggle Titanic 生存预测 -- 详细流程吐血梳理](https://zhuanlan.zhihu.com/p/31743196)
