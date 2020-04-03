---
title: testing
date: 2020-13-03

# Put any other Academic metadata here...
summary: "testing 12 3"
[[main]]
  name = "CV"
  url = "files/cv.pdf"
  weight = 70
---



```python
#
# Data Mining
# Assignment
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
   
plt.show()

```
