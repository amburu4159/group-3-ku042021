# Segment 1
## Purpose
  The purpose of this investigation is to determine the primary factors that contribute to the fatality of vehicle crashes. The data for this investigation is pulled from the NHTSA CRSS (National Highway Traffic Safety Administration Crash Report Sampling System) database (https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system). 
## Teammates
1. Andrew Mburu - Square/Repository
2. Bryan Gurss - Triangle/ML model  
3. Jason Dibble - Circle/Database  
4.  Darpan Bhakta - X/Technologies
 
## Questions to address
  o What factors contribute to fatality in vehicle crashes?  
  o Do accidents involving alcohol or distracted driving have increased rates of fatality?  
  o Does adverse weather impact the rate of crash fatality?
  
  
  ## slides draft 
https://docs.google.com/presentation/d/1Jm8NRDJl4Hmu8E_V1p_JUmdTqQMMCMX8bRXLLxojHeE/edit?usp=sharing


## Communication Protocols
For communication purposes, we are communicating using slack and zoom meetings. We message each other regularly have had a handful of meetings outside of class to discuss the project, potential road-blocks, update each other on the status of our individual parts and next steps.

## Resources
NHTSA website (https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/CRSS/2019/)

Google docs

## Dependencies
o import numpy as np
o import pandas as pd

o from pathlib import Path

o from collections import Counter

o from matplotlib import pyplot as plt

o from sklearn.model_selection import train_test_split

o from sklearn.preprocessing import StandardScaler

o from sklearn.ensemble import RandomForestClassifier

o from sklearn.feature_selection import SelectFromModel

o from imblearn.over_sampling import SMOTE

o from sklearn.linear_model import LogisticRegression

o from sklearn.metrics import balanced_accuracy_score

o from sklearn.metrics import confusion_matrix

o from imblearn.metrics import classification_report_imbalanced

## Technology at use
  The database will be setup using postgreSQL. Python is the primary language that will be used for data manipulation and exploration. Supervised machine learning will be used to make predictions, and Tableau will be used to visualize the data.
  
## ERD
 ![ERD Diagram](https://user-images.githubusercontent.com/40553064/133019580-46009ee0-8d99-48b4-9e23-f882a34a61cf.PNG)
 
 ## Database
The database being used is AWS RDS with access and manipulation via postgreSQL. The Data being used is from the NHTSA (National Highway and Traffic Safety Administration) government site (www.nhtsa.gov). The csvs are downloaded and accessed via jupyter notebook to pre-process the data. The tables are then loaded into the SQL database using sqlalchemy, and two tables (vehicle and distraction) are combined within postgres for ease of access later. The data is now accessible from the database for the machine learning model.

 
 ## Machine Learning Algorithm / Inital Data Analysis
We began the learning model by creating our features and target variables.  The “MAX_SEVNAME” column will be used as our target variable y, while all other columns in the dataframe will be used as our features (X).  From here the following things will be performed on the data.
1. The data is split into testing and training groups
2. The data is scaled to make all data equally important
3. A Random Forest Classifier is used on the data to determine the importance of each feature.  The training data group is used to determine this importance.
4. A horizontal bar graph is presented with the level of importance in descending order.
5. The data is then narrowed down to only the most important features determined by the model
6. A new training and testing set is created with the only the important features
7. The data is put into the SMOTE trainer
8. The data is fitted to a logistic regression model
9. An accuracy score, confusion matrix, and a classification report are developed




