---
title: Assignment 2
date: 2020-02-04

# Put any other Academic metadata here...
summary: "The goal of this assignment is to learn about kNN on the IRIS dataset."
---

{{% staticref "files/Assignment 2.ipynb" "newtab" %}}Please download my ipynb code here.{{% /staticref %}}

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
            # is like -2.220446049250313e-16 which is almost 0, but not 0
            # this if statement just catches and corrects those instances
            if measure < 0: measure = 0.0
            
        return measure

    
    
    def Compute_Distances_toOtherRecords(self, distance_metric, unknown_record):
        
        distances = []
        for i in range(self.labled_records.shape[0]):
            # gets record with class label
            labled_record_i = self.Get_Record(self.labled_records, i)
               
            # strips class label for record comparison
            record_i = self.Split_Record_fromClass(labled_record_i)[0]
            
            # actual record comparison
            distance = self.Calculate_Distance(distance_metric, record_i, unknown_record)
            
            # measure recorded 
            distances.append((distance, labled_record_i))
                
        return distances



    def Identify_kNNs(self, k, distances):
    
        # selects the kNNs
        distances.sort(key = lambda x: x[0]); NNs = distances[:k]
        
        # strips the no longer needed distance value
        for index, nn in enumerate(NNs): NNs[index] = nn[1][0][:]
    
        return NNs



    def Determine_ClassLabel_ofUnknownRecord(self, kNNs):
    
        # puts together a vote counting mechanism
        votes = {}
        for knn in kNNs:
            if knn[-1] not in votes.keys(): votes[knn[-1]] = 1
            else: votes[knn[-1]] += 1
        votes = [(vote, votes[vote]) for vote in votes]
        
        # selects majority vote
        majority_vote = max(votes, key=lambda x: x[1])[0]
        
        return majority_vote

    
    
    def Classify_UnknownRecord(self, distance_metric, k, unknown_record):
        
        # follows the algorithm described in class
        
        # 1st calculate distances
        distances = self.Compute_Distances_toOtherRecords(distance_metric, unknown_record)
        
        # 2nd identify kNNs
        kNNs = self.Identify_kNNs(k, distances)     
        
        # 3rd determine class label
        classified_label = self.Determine_ClassLabel_ofUnknownRecord(kNNs)
    
        return classified_label
    
    
    
    def Score_Config(self, config, test_records=None):
        
        if isinstance(test_records, np.ndarray) == False: test_records = self.labled_records.copy()
        else: test_records = test_records.copy()
        
        distance_metric, k = config; correct_count = 0; cap = test_records.shape[0]
        for i in range(cap):
            test_records_copy = test_records.copy()
            
            # creates an unknown record from a known record
            test_record_i = self.Get_Record(test_records, i)
            
            # creates a test_records that does not contain the searched record
            if i == 0: test_records = test_records[1:]
            elif i == cap-1: test_records = test_records[:-1]
            else: test_records = np.delete(test_records, [i], axis=0)
            
            unknown_record, actual_class_label = self.Split_Record_fromClass(test_record_i)
            
            # makes classification
            classified_label = self.Classify_UnknownRecord(distance_metric, k, unknown_record)
            test_records = test_records_copy.copy()
            
            # adjusts score
            if classified_label == actual_class_label: correct_count += 1
                
        score = correct_count / cap
    
        return score
    
    
    
    def Train(self, labled_records):
        
        # saves a copy of the labled records
        self.labled_records = labled_records.copy()
    
        # tests different configurations
        configs = self.Get_Configs()
        results = [(self.Score_Config(config), config) for config in configs]
        self.ShowAccuracy(results)
    
        # hyperparameters tuning
        results.sort(key=lambda x: x[0], reverse=True)
        optimal_hyperparameters = results.pop(0)[1]
        while optimal_hyperparameters[1] == 1: optimal_hyperparameters = results.pop(0)[1]
    
        self.optimal_hyperparameters = optimal_hyperparameters
                
        return

    
    
    def ShowAccuracy(self, results):
        
        if isinstance(results, list) == False:
            test_records = results
            results = [(self.Score_Config(self.optimal_hyperparameters, test_records), self.optimal_hyperparameters)]
        
        bins = {}
        for result in results:
            k = result[1][1]
            if k not in bins: bins[k] = [result]
            else: bins[k].append(result)
    
        # for each k value it displays a bar graph
        k_values = list(bins.keys())
        for k in k_values:
        
            figure = plt.figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k').add_subplot()        
            figure.set_xlabel('k = ' + str(k))
        
            score = 0
            dms = [] # creates the bars
            for element in bins[k]:
                score = element[0]
                dm = element[1][0]
                dms.append(dm)
                plt.bar(dm, score)
        
            # when performing training accuracy
            if len(k_values) > 1:
                figure.set_xticklabels(dms, rotation=10)     
                figure.set_title('Holding k fixed at: ' + str(k))
                figure.set_ylabel('Accuracy')
        
            # when performing final accuracy
            if len(k_values) == 1:
                figure.set_title('Optimal Hyperparameters: ' + dms[0] + ', k = ' + str(k))
                figure.set_ylabel('Final Accuracy')
                figure.text(-.25, 0, 'Final Accuracy: ' + '{:2.2f}'.format(score*100) + '%', verticalalignment='bottom', color='black', fontsize=15)
        
        return
```


```python
# here we create the path to the location of our iris.data file
iris_data_path = path[0] + '/data/iris.data'

# now we create our 2 data sets
development_data, test_data = Get_Iris_Development_andTest_Data(iris_data_path)

# next we instantiate our kNN Classifier
knn_classifier = kNN_Classifier()
knn_classifier.Train(development_data)
knn_classifier.ShowAccuracy(test_data)

plt.show()
```



{{< figure library="true" src="a2_output_0_0.jpg" lightbox="true" >}}



{{< figure library="true" src="a2_output_0_1.jpg" lightbox="true" >}}



{{< figure library="true" src="a2_output_0_2.jpg" lightbox="true" >}}



{{< figure library="true" src="a2_output_0_3.jpg" lightbox="true" >}}



{{< figure library="true" src="a2_output_0_4.jpg" lightbox="true" >}}



```python

```
