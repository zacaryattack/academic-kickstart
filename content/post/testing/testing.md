---
title: testing
date: 2020-13-03

# Put any other Academic metadata here...
summary: "testing 12 3"
---



```python
#
# Zacary Cotton
# 100 103 1997
# CSE 5334 - Data Mining
# Assignment 2
#

from sys import path # to locate the csv
import pandas as pd # to load the csv
import numpy as np # to utilize array structures
import matplotlib.pyplot as plt # to display results
```


```python
# the first necessary function would be that which retrieves the data
def Get_Iris_Development_andTest_Data(iris_data_path, development_percentage=0.75):
    
    # first thing, read the csv with the iris data
    iris_data_df = pd.read_csv(iris_data_path)
    
    # add colume names to give data meaning
    iris_data_df.columns = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class']
        
    # shuffles the dataframe & resets the its index
    iris_data_df = iris_data_df.sample(frac=1).reset_index(drop=True)
    
    # change all flower string names to enumerated values
    flower_names = list(set(iris_data_df.iloc[:, -1]))
    for id, name in enumerate(flower_names): iris_data_df['class'] = iris_data_df['class'].str.replace(name, str(id))
    iris_data_df['class'] = pd.to_numeric(iris_data_df['class'])
    
    # shuffles the dataframe & resets the its index
    iris_data_df = iris_data_df.sample(frac=1).reset_index(drop=True)
        
    # splits the data here
    cap = int(iris_data_df.shape[0] * development_percentage)
    development = iris_data_df[:cap]
    test = iris_data_df[cap:]
    
    # shuffles the dataframe & resets the its index
    development = development.sample(frac=1).reset_index(drop=True)
    
    # shuffles the dataframe & resets the its index
    test = test.sample(frac=1).reset_index(drop=True)
    
    development = development.values
    test = test.values
        
    return development, test
```


```python
# the next step would be to design a kNN Classifier
class kNN_Classifier:
    
    def __init__(self):
        
        # the learned data
        self.labled_records = None
        self.optimal_hyperparameters = None
        
        # helper function
        def Get_Record(Record_Matrix, record_number):
            row = Record_Matrix[record_number, :]
            record = row.reshape(1, row.shape[0])
            return record
        self.Get_Record = Get_Record

        # helper function
        def Split_Record_fromClass(record):
            records_class = record[0][-1]
            record = record[:, :-1]
            return record, records_class
        self.Split_Record_fromClass = Split_Record_fromClass
        
        # helper function
        def Get_Configs():
            distance_metrics = ['Euclidean Distance', 'Normalized Euclidean Distance', 'Cosine Similarity']
            k_values = [1, 3, 5, 7]
            
            configs = []
            for i in distance_metrics:
                for j in k_values:
                    config = [i, j]
                    configs.append(config)
            return configs
        self.Get_Configs = Get_Configs
        
        return

    
    
    def Calculate_Distance(self, distance_metric, record1, record2):
    
        def Calculate_Norm(v): return sum([value**2 for value in v[0]])**.5
        
        
        # calculates Euclidean Distance
        if distance_metric.lower() == 'Euclidean Distance'.lower():                                
            measure = [(value - record2[0][index])**2 for index, value in enumerate(record1[0])]            
            measure = sum(measure)**.5
            
        # calculates Normalized Euclidean Distance
        elif distance_metric.lower() == 'Normalized Euclidean Distance'.lower():
            record1_average = np.average(record1)
            record2_average = np.average(record2)  
            
            numerator = (Calculate_Norm((record1 - record1_average) - (record2 - record2_average)))**2            
            denominator = (2*(((Calculate_Norm(record1 - record1_average))**2) + ((Calculate_Norm(record2 - record2_average))**2)))
            
            measure = numerator / denominator
        
        # calculates Cosine Similarity
        elif distance_metric.lower() == 'Cosine Similarity'.lower():
            numerator = sum([(value * record2[0][index]) for index, value in enumerate(record1[0])])             
            norm_record1 = Calculate_Norm(record1)
            norm_record2 = Calculate_Norm(record2)
            denominator = norm_record1 * norm_record2
            
            # the value is subtracted from 1 so that no special care need be taken when comparing
            # this measure against a measure calculated from the other distance metrics
            measure = 1 - (numerator / denominator)
            
            # measure = 1 - (numerator / denominator) should always be a positive value
            # however due to floating point calculations in python, sometimes this number
           

    
    
   
plt.show()

```
