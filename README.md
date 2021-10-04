# Car Fatalities
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
  
  
  ## Google Slides Draft 
https://docs.google.com/presentation/d/1Jm8NRDJl4Hmu8E_V1p_JUmdTqQMMCMX8bRXLLxojHeE/edit?usp=sharing


## Resources
NHTSA website (https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/CRSS/2019/)
FARS data (https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/FARS/)

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

 ## Machine Learning Algorithm / Inital Data Analysis
We began the learning model by creating our features and target variables. The “MAX_SEVNAME” column will be used as our target variable y, while all other columns in the dataframe will be used as our features (X). From here the following things will be performed on the data.  The “MAX_SEVNAME” name will be changed to “Number of Deaths”
 
### Preliminary Data Preprocessing
We are storing all of our data on the AWS servers.  Four different csv files are pulled for processing before running all of the data through the machine learning model.  Here are some actions that took place with the data:
- The first three dataframes were merged together. 
- Duplicate and Null rows were dropped. 
- We took out all data that did not relate to the driver of the vehicles involved (Known as person 1 in the data)
- Rows that included any deaths were removed from the dataset.  This was done because this data would also appear in the fatalities data that will be added later.
- We added the fatalities data onto the end of the original dataframe.  This created a large dataset that included accidents with and without fatalities
- We removed any rows that had information that would not be conducive to the machine learning model.  
  - Speed over 900 mph
  - Car years over 2025
  - Ages over 120
  - Hour of Day over 25
After we got the data to look like what we wanted, we used the OneHotEncoder to make all object variables into numbers (zeros and ones).  After this we used the standard scaler in order to scale down some of the other variables. This was done to keep features with large numbers from skewing the machine learning model.  Once the data was scaled the following steps took place.

1. The data is split into testing and training groups.  We used the standard 80/20 split that is custom in many models
2. The data is scaled to make all data equally important
3. A Random Forest Classifier is used on the data to determine the importance of each feature. The training data group is used to determine this importance.
4. A horizontal bar graph is presented with the level of importance in descending order.  This graph was created to help with our presentation.  We used the most important features table to help make decisions about which features to draw attention to when presenting.
5. The data was initially narrowed down to only those important features, but was later expanded because of better results in the prediction algorithm
6. The data is fitted to a logistic regression model (specifically into the AdaBoostClassifier).  The SMOTE random sampler was originally used because the dataset that we were working with had few examples of fatal accidents, which made it tougher for the machine learning model to successfully perform its task.  Recently, the group decided to add in a fatality dataset.  This would create more data with fatalities involved and would probably allow us to choose a different model than the SMOTE oversampler.  The integration of the newer dataset did give us different important features and much higher accuracy, precision, and recall from the SMOTE used in the original model.  We also tried to use a deep neural network, however, the model presented challenges of overfitting.
8. An accuracy score, confusion matrix, and a classification report are developed from this information.

### Accuracy Report
The accuracy report with descriptions are shown below:

![AccuracyInfo.png](AccuracyInfo.png)

The table gives an accuracy score approximately 84%.  This means that the prediction algorithm was correct 84% of the time when run on the testing dataset.  This is not perfect but is a good score for the situation.

The confusion matrix gives the following information:
- The precision score of .95 tells us that when the model predicted there would be no death, it was correct 95% of the time.
- The precision score of .83 tells us that when the model predicted there would be death, it was correct 83% of the time.
- The recall/sensitivity score of .98 tells us that when the model predicted the non fatal accidents correctly 98% of the time.
- The recall/sensitivity score of .70 tells us that when the model predicted the fatal accidents correctly 70% of the time.

Overall, these scores tell us that it is plausible for us to make predictions on fatal accidents.  This will also ultimately tell us there are steps we can take to improve driver safety.



## Technology at use
  The database will be setup using postgreSQL. Python is the primary language that will be used for data manipulation and exploration. Supervised machine learning will be used to make predictions, and Tableau will be used to visualize the data.
  
## ERD
 ![ERD Diagram](https://user-images.githubusercontent.com/40553064/133019580-46009ee0-8d99-48b4-9e23-f882a34a61cf.PNG)
 
 ## Database
The database being used is AWS RDS with access and manipulation via postgreSQL. The Data being used is from the NHTSA (National Highway and Traffic Safety Administration) government site (www.nhtsa.gov). The csvs are downloaded and accessed via jupyter notebook to pre-process the data. The tables are then loaded into the SQL database using sqlalchemy, and two tables (vehicle and distraction) are combined within postgres for ease of access later. The data is now accessible from the database for the machine learning model.


## Dashboard
Here is a link to the Tableau Dashboard

https://public.tableau.com/app/profile/andrew.mburu8003/viz/ku_data_project3/FatalityAccidents?publish=yes




