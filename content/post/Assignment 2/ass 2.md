---
title: Ovssion
date: 2020-13-04

# Put any other Academic metadata here...
summary: "Demonstraomial Regression."
---


```python
#
# Zacary Cotton
# 100 103 1997
# CSE 5334 - Data Mining
# Assignment 1
#

import math, random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge




# Uses the uniform distribution between 0 and 1 for generating X values
def Generate_Xs(count):
    
    Xs = []
    for i in range(count):
        x = random.uniform(0, 1)
        Xs.append(x)
    Xs = np.sort(np.asarray(Xs))
    
    return Xs




# Generates Y values using y = sin(2*pi*X) + N
def Generate_Ys(Xs):
    
    Y = []
    for x in Xs:
        # Samples N from the normal gaussian distribution
        N = random.gauss(0, 1)
        y = math.sin(2*math.pi*x) + N
        Y.append(y)
    Y = np.asarray(Y)
    
    return Y




# A helper class for using matplotlib to draw graphs predictions vs ground truth
class Figure_Prediction_vs_Truth:
    
    def __init__(self, subplot_row_count, x_plot, y_plot, x_train, y_train):
        
        self.x_plot = x_plot
        self.y_plot = y_plot
        
        self.x_train = x_train
        self.y_train = y_train
        
        self.subplot_row_count = subplot_row_count
        self.subplot_empty_spot = 0
        
        if self.subplot_row_count == 2: figure_size = (12, 10)
        elif self.subplot_row_count == 3: figure_size = (12, 14)
        else: figure_size = (12, 4)
        
        self.figure = plt.figure(num=None, figsize=figure_size, dpi=80, facecolor='w', edgecolor='k')
        
        return
    
    
    
    def Add_Subplot(self, title, y_pred):
        
        self.subplot_empty_spot += 1
        subplot = self.figure.add_subplot(self.subplot_row_count, 2, self.subplot_empty_spot)
        
        subplot.text(.75, max(self.y_train), title, style='italic', fontsize=12)
        subplot.axis([0, 1, min(self.y_train)-.55, max(self.y_train)+.55])
        
        subplot.plot(self.x_plot, self.y_plot, color='lime', linewidth=2)
        
        subplot.scatter(self.x_train, self.y_train, s=70, facecolors='none', edgecolors='blue', marker='o')
    
        subplot.plot(self.x_plot, y_pred, color='red', linewidth=2)
        
        return




# Uses root mean square error to find weights of polynomial regression
def RMSE_toFind_Weights_ofRegression(chart_orders, x_plot, y_plot, x_train, y_train, x_test, y_test):
    
    weights = []; rmse_train = []; rmse_test = []
    Degree_Figure = Figure_Prediction_vs_Truth(2, x_plot, y_plot, x_train, y_train)
    for degree in range(10):
        
        model = make_pipeline(StandardScaler(), PolynomialFeatures(degree), Ridge(alpha=0, fit_intercept=True, solver='svd'))
        model.fit(x_train[:, np.newaxis], y_train)
        
        y_pred = model.predict(x_train[:, np.newaxis])
        rmse_train.append(math.sqrt(mean_squared_error(y_train, y_pred)))    
        
        y_pred = model.predict(x_plot[:, np.newaxis])
        w = [model.steps[2][1].intercept_.tolist()]
        w.extend(model.steps[2][1].coef_[1:].tolist())
        weights.append((degree, w))
        
        if degree in chart_orders: Degree_Figure.Add_Subplot('M = ' + str(degree), y_pred)
        
        if degree == 9: Ninth_Order_Prediction = y_pred
        
        y_pred = model.predict(x_test[:, np.newaxis])
        rmse_test.append(math.sqrt(mean_squared_error(y_test, y_pred)))
    
    return weights, rmse_train, rmse_test, Ninth_Order_Prediction




# Creates the coefficient table for viewing the weights and biases
def Display_WeightsTable(print_orders, weights):
    
    p_str = ''
    
    print_weights = []
    for weight in weights:
        if weight[0] in print_orders:
            if weight[0] == print_orders[-1]: cap = len(weight[1])
            if weight[0] == print_orders[0]: p_str += ' '*8
            else: p_str += ' '*5
            p_str += ('M = ' + str(weight[0]))
            if weight[0] == print_orders[1]: weight[1][0] += 1
            print_weights.append(weight[1])
    
    p_str += ('\n' + '-'*(len(p_str)+4) + '\n')
    
    for i in range(cap):
        w_spacing = ('w' + str(i) + '|')
        p_str += w_spacing
        w_spacing = len(w_spacing)
        for index, weight in enumerate(print_weights):
            try:
                fi = 7; fd = 3; max_len = fi + fd
                f_str = '{:' + str(fi) + '.' + str(fd) + 'f}'
                w = f_str.format(weight[i])
                
                while len(w) > max_len-1 and fd > 0:
                    fd -= 1; f_str = '{:' + str(fi) + '.' + str(fd) + 'f}'
                    w = f_str.format(weight[i])
                
                if index == 0: p_str += ' '*w_spacing
                elif len(w) > 6: p_str += ' '*w_spacing
                else: p_str += ' '*4
                
                p_str += w
                
            except: p_str += ' '*10
        p_str += '\n'
    
    print(p_str)
    
    return




# A helper function for using matplotlib to draw graphs for error
def Draw_TrainError_vs_TestError(test_type, rms_train, rms_test):
    
    rms_train = [rms/max(rms_test) for rms in rms_train]
    rms_test = [rms/max(rms_test) for rms in rms_test]
    
    figure = plt.figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k').add_subplot()
    figure.set_xlabel(test_type); figure.set_ylabel('RMS')
    
    figure.plot(range(len(rms_train)), rms_train, color='blue', linewidth=2, label='Training')
    figure.scatter(range(len(rms_train)), rms_train, s=70, facecolors='none', edgecolors='blue', marker='o')
    figure.plot(range(len(rms_test)), rms_test, color='red', linewidth=2, label='Test')
    figure.scatter(range(len(rms_test)), rms_test, s=70, facecolors='none', edgecolors='red', marker='o')
    
    if test_type == 'M':
        figure.set_xticks(np.arange(0, 10, 3))
        figure.set_yticks(np.arange(0.0, 1.2, .2))
        figure.legend(loc='upper left')
    
    elif test_type == 'λ':
        plt.xticks(range(len(rms_test)), ['0','1','1/10','1/100','1/1000','1/10000'])
        figure.set_yticks(np.arange(0.0, 1.2, .2))
        figure.legend(loc='upper right')
    
    return




# Regularizes using the sum of weights and sklearn's Ridge function
def Regularize_Using_Sum_ofWeights(Lambdas, x_plot, y_plot, x_train, y_train, x_test, y_test):
    
    rmse_train = []; rmse_test = []
    Lambda_Figure = Figure_Prediction_vs_Truth(3, x_plot, y_plot, x_train, y_train)
    for Lambda in Lambdas:
        L = Lambda
        if Lambda > 0: Lambda = 1.0/Lambda
    
        model = make_pipeline(StandardScaler(), PolynomialFeatures(9), Ridge(alpha=Lambda, fit_intercept=True, solver='svd'))
        model.fit(x_train[:, np.newaxis], y_train)
        
        y_pred = model.predict(x_train[:, np.newaxis])
        rmse_train.append(math.sqrt(mean_squared_error(y_train, y_pred)))
    
        y_pred = model.predict(x_plot[:, np.newaxis])
        Lambda_Figure.Add_Subplot('λ = ' + str(L), y_pred)
    
        y_pred = model.predict(x_test[:, np.newaxis])
        rmse_test.append(math.sqrt(mean_squared_error(y_test, y_pred)))
    
    return rmse_train, rmse_test




# Generates 20 data pairs (X, Y) using y = sin(2*pi*X) + N
# plus another 100 pairs used exclusively for plotting with matplotlib
x_plot = Generate_Xs(100); y_plot = Generate_Ys(x_plot)

# Uses 10 for train and 10 for test
x_train = Generate_Xs(10); y_train = Generate_Ys(x_train)
x_test = Generate_Xs(10); y_test = Generate_Ys(x_test)

chart_orders = [0, 1, 3, 9]
# Uses root mean square error to find weights of polynomial regression
weights, rmse_train, rmse_test, Ninth_Order_Prediction = RMSE_toFind_Weights_ofRegression(chart_orders, x_plot, y_plot, x_train, y_train, x_test, y_test)

print_orders = [0, 1, 6, 9]
Display_WeightsTable(print_orders, weights)

# Draws train error vs test error
Draw_TrainError_vs_TestError('M', rmse_train, rmse_test)

# Generates 100 more data and fits to the 9th order model and then draws the fit
_100_More_Data_x = Generate_Xs(100); _100_More_Data_y = Generate_Ys(_100_More_Data_x)
Figure_Prediction_vs_Truth(1, x_plot, y_plot, _100_More_Data_x, _100_More_Data_y).Add_Subplot('N = 100', Ninth_Order_Prediction)

Lambdas = [0, 1, 10, 100, 1000, 10000]
# Regularizes using the sum of weights
rmse_train, rmse_test = Regularize_Using_Sum_ofWeights(Lambdas, x_plot, y_plot, x_train, y_train, x_test, y_test)

# Draws test and train error according to lamda
Draw_TrainError_vs_TestError('λ', rmse_train, rmse_test)

plt.show()

p_str = '\n'
p_str += '     When M is order 3, the model consistently produces a function that\n'
p_str += 'closely follows the inner, more general shape of the data\'s true form,\n'
p_str += 'while when M is order 9, it models the Train Data perfectly, thus over-\n'
p_str += 'fitting is achieved.\n'
p_str += '\n'
p_str += '     When λ is 0 the model behaves very much like an ordinary regression\n'
p_str += 'with order 9, but when λ is from 1 to 100, it often produces more general\n'
p_str += 'shapes of the data\'s true form, λ at 10 appears to often fit the best,\n'
p_str += 'but sometimes 1 fits better. However, with λ above 100 it appears to begin\n'
p_str += 'to perfectly fit the Train Data yet again.\n'
p_str += '\n'
p_str += '     Based on the best test performance for M only, a model with M = 5 or 6\n'
p_str += 'often produces the best results as M = 9 overfits. Based on the best test\n'
p_str += 'performance for when M = 9 and λ varies, when λ = 10, it often produces the\n'
p_str += 'better fit however, when λ = 100 it also often appears to best match the data\'s\n'
p_str += 'true form.\n\n\n'

print(p_str)

```

            M = 0     M = 1     M = 6     M = 9
    -----------------------------------------------
    w0|     0.013     1.013     0.719   7189.746
    w1|              -0.604     6.928   141814.38
    w2|                        -1.342   -429087.1
    w3|                       -59.989   -2728981
    w4|                       -51.976   5907832.3
    w5|                         8.740   14328446
    w6|                        11.659   -18756249
    w7|                                 -27228921
    w8|                                 1902265.4
    w9|                                 5575594.7
    



{{< figure library="true" src="output_0_1.jpg" lightbox="true" >}}



{{< figure library="true" src="output_0_2.jpg" lightbox="true" >}}



{{< figure library="true" src="output_0_3.jpg" lightbox="true" >}}



{{< figure library="true" src="output_0_4.jpg" lightbox="true" >}}



{{< figure library="true" src="output_0_5.jpg" lightbox="true" >}}



         When M is order 3, the model consistently produces a function that
    closely follows the inner, more general shape of the data's true form,
    while when M is order 9, it models the Train Data perfectly, thus over-
    fitting is achieved.
    
         When λ is 0 the model behaves very much like an ordinary regression
    with order 9, but when λ is from 1 to 100, it often produces more general
    shapes of the data's true form, λ at 10 appears to often fit the best,
    but sometimes 1 fits better. However, with λ above 100 it appears to begin
    to perfectly fit the Train Data yet again.
    
         Based on the best test performance for M only, a model with M = 5 or 6
    often produces the best results as M = 9 overfits. Based on the best test
    performance for when M = 9 and λ varies, when λ = 10, it often produces the
    better fit however, when λ = 100 it also often appears to best match the data's
    true form.
    
    
    



```python

```
