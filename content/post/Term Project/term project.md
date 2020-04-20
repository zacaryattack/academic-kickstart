---
title: Term Project - CSE 5334 - Data Mining
date: 2020-01-05

# Put any other Academic metadata here...
summary: "The goal of this project is to develop a classifier to aid in the classification of comments onto a rating scale of 1 to 10."
---

#{{% staticref "files/100 103 1997 Zacary Cotton - Term Project.ipynb" "newtab" %}}Please download my ipynb code here.{{% /staticref %}}

To successfully execute this .ipynb file, the dirrectory 'boardgamegeek-reviews'
must be located within the same dirrectory as this .ipynb file.The boardgamegeek-reviews dirrectory can be obtained from url:In this blog post we will develop a classifier to aid in the classification of
comments onto a rating scale of 1 to 10 values.We begin by importing the necessary libraries and building the csv path in
order to load the data, which upon download is located in a dirrectory called
boardgamegeek-reviews, in csv file named 'bgg-13m-reviews.csv'.

```python
from sys import path
from os.path import join

import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score




csv_path = join(path[0], 'boardgamegeek-reviews')
csv_path = join(csv_path, 'bgg-13m-reviews.csv')

print(csv_path)
```

    /Users/zmo/here/jupyter/data mining/project/boardgamegeek-reviews/bgg-13m-reviews.csv

Now that we have a path to our data we must read it into memory and preprocess
it by formating the comments and ratings. For the comments, we format off
unnecessary elements such as extra whitespace, non-alpha characters, non-english
characters, and make all characters lowercase. For the ratings, we simply ensure
that all ratings have values that are type float.

For this we create 2 helper functions, Load_Data() and Format_DF().

```python
def Format_DF(df):
        
    def Has_Letters(in_str): return in_str.lower().islower()

    
    def Is_RomanAlphabet(in_str):
        try: in_str.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError: return False
        return True
    
    
    # only keeps 'comment' and 'rating' columns, filling NaNs with ''
    df = df[['comment', 'rating']].copy().fillna('')
    # only keeps items that actually have a comment
    df = df[df['comment'].apply(lambda x: len(x) > 0)]
        
    # strips and lowers comments
    df['comment'] = [comment.strip().lower() for comment in df['comment']]
    # only keeps items that actually have a comment
    df = df[df['comment'].apply(lambda x: len(x) > 0)]
        
    # filters out comments that do not contian letters
    df['comment'] = [comment if Has_Letters(comment) == True else '' for comment in df['comment']]
    # only keeps items that actually have a comment
    df = df[df['comment'].apply(lambda x: len(x) > 0)]

    # filters out comments that are not Roman Alphabet comprised
    df['comment'] = [comment if Is_RomanAlphabet(comment) == True else '' for comment in df['comment']]
    # only keeps items that actually have a comment
    df = df[df['comment'].apply(lambda x: len(x) > 0)]
    
    # light format of ratings to floats
    df['rating'] = [float(rating) for rating in df['rating']]
    
    # shuffles the df
    df = df.sample(frac=1).reset_index(drop=True)
        
    return df




def Load_Data(csv_path):
    
    df = pd.read_csv(csv_path)
    
    df = Format_DF(df)
    
    return df




all_data_df = Load_Data(csv_path)
```
Our data has been fully loaded and has been preprocessed. We may now take
a little look at the data that we are dealing with in this project.

```python
print(all_data_df)
```

                                                       comment  rating
    0        exploration and race game combined.  entertain...    8.00
    1        the learning curve can be high for some but on...    9.00
    2        lots of terrain and fun figures to start out w...   10.00
    3        surprisingly good light game. plays very fast ...    6.00
    4        good game for some friends, especially when th...    4.00
    ...                                                    ...     ...
    2482040                                                bga    7.00
    2482041  one of my favourites. it's always a challenge ...    9.14
    2482042                             it's an rpg board game    7.20
    2482043  tta is a game that demands your attention and ...    8.00
    2482044  this game is way too luck based for my likes. ...    3.00
    
    [2482045 rows x 2 columns]

Next we must visualize our data in different forms in order to spot any useful
elements that could aid us in the development of a classification algorithm.

For this we create another helper function, Get_DF_byClass().

```python
def Get_DF_byClass(df):
    
    df_by_class = {}
    for rating in set([int(rating) for rating in df['rating']]):
        
        df_n = df.loc[df['rating'] >= ((rating-1)+.5)]
        df_n = df_n.loc[df_n['rating'] <= (rating+.5)]
        df_n = df_n.sample(frac=1).reset_index(drop=True)
        
        df_by_class[rating] = df_n
    
    return df_by_class




df_by_class = Get_DF_byClass(all_data_df)

total = 0
for rating in df_by_class:
    print('   rating: ' + str(rating), 'count: ' + str(df_by_class[rating].shape[0]))
    total += df_by_class[rating].shape[0]

print('\ntotal data count: ', total)
```

       rating: 0 count: 10
       rating: 1 count: 20077
       rating: 2 count: 38921
       rating: 3 count: 74425
       rating: 4 count: 130362
       rating: 5 count: 242860
       rating: 6 count: 496911
       rating: 7 count: 726467
       rating: 8 count: 614929
       rating: 9 count: 301233
       rating: 10 count: 144029
    
    total data count:  2790224

The print out above is of all ratings grouped together into integer range groups.
These ranges were created by asigning a rating to a group by following this
simple rule,
            if rating - int(rating) <= .5
            then add rating to group int(rating)

From the print out above, it can be seen that one range is extremely underrepresented.

Let's see our rating grouping by bar chart.

```python
figure = plt.figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k').add_subplot()        
figure.set_xlabel('Ratings')
figure.set_title('Ratings & Their Counts')
figure.set_ylabel('Rating Counts')
figure.set_xticks(np.arange(0, 11, 1))
for rating in df_by_class: plt.bar(rating, df_by_class[rating].shape[0])
```


{{< figure library="true" src="p1.jpg" lightbox="true" >}}

The range of ratings around 0 is equal to 10. That is, our smallest class is
represented by only 10 elements whereas our most dominant class is represented
by around 700000 elements. Becuase of the drastic difference in representation,
we will be dropping data records where rating is < 1.

```python
all_data_df = all_data_df.loc[all_data_df['rating'] >= 1]

df_by_class = Get_DF_byClass(all_data_df)
```
Let's see our rating grouping by bar chart again now that we have eliminated
the outlier class of rating values < 1

```python
figure = plt.figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k').add_subplot()        
figure.set_xlabel('Ratings')
figure.set_title('Ratings & Their Counts')
figure.set_ylabel('Rating Counts')
figure.set_xticks(np.arange(0, 11, 1))
for rating in df_by_class: plt.bar(rating, df_by_class[rating].shape[0])
```


{{< figure library="true" src="p2.jpg" lightbox="true" >}}



```python
print('\nafter removing classes less than 1')
total = 0
for rating in df_by_class:
    print('   rating: ' + str(rating), 'count: ' + str(df_by_class[rating].shape[0]))
    total += df_by_class[rating].shape[0]

print('\ntotal data count: ', total, '\n')
```

    after removing classes less than 1
       rating: 1 count: 20073
       rating: 2 count: 38921
       rating: 3 count: 74425
       rating: 4 count: 130362
       rating: 5 count: 242860
       rating: 6 count: 496911
       rating: 7 count: 726467
       rating: 8 count: 614929
       rating: 9 count: 301233
       rating: 10 count: 144029
    
    total data count:  2790210

As can be seen in the print out above, we have enough elements for each class
that we can draw a smaller smaple of the data to use during classification
algorithm development and then later we can test our algorithm on the whole set.

So now we create another helper function, Get_Scaled_Data()

```python
def Get_Scaled_Data(df, N):
    
    df_by_class = Get_DF_byClass(df)
    
    total_size = 0
    for key in df_by_class: total_size += df_by_class[key].shape[0]
        
    scaled_df = []; scale = N / total_size
    for rating in df_by_class:
        
        i = int(df_by_class[rating].shape[0] * scale)
        scaled_df.append(df_by_class[rating][0:i])
        
    scaled_df = pd.concat(scaled_df).sample(frac=1).reset_index(drop=True)
    
    return scaled_df




N = 500000
scaled_df = Get_Scaled_Data(all_data_df, N)

df_by_class = Get_DF_byClass(scaled_df)

total = 0; print()
for rating in df_by_class:
    print('   rating: ' + str(rating), 'count: ' + str(df_by_class[rating].shape[0]))
    total += df_by_class[rating].shape[0]

print('\ntotal data count: ', scaled_df.shape[0])
```

       rating: 1 count: 3753
       rating: 2 count: 7533
       rating: 3 count: 14548
       rating: 4 count: 25732
       rating: 5 count: 49883
       rating: 6 count: 108142
       rating: 7 count: 163578
       rating: 8 count: 139790
       rating: 9 count: 67948
       rating: 10 count: 29212
    
    total data count:  499996

From the print out above it can be seen that now our smallest class representation
is closer to 3,000 and our largest class representation is closer to 160000.

Get_Scaled_Data simply scalled each group so that the count of all ratings would
be less than or equal to N, or 500,000

Our data count is now 499996, this should be large enough to develop our algorithm
while also being small enough that algorithm test times will not take longer than
30 minutes to compute.

So now we must split our dataframe into the x_train, y_train, x_dev, y_dev,
and x_test, y_test data sets. However, becuase each x value is a string we will
need to convert each x value into a numerical sequence by using a vectorizer.
This vectorizer will be needed anytime we want to pass our classifier a string
to classify. So to perform these actions we will create another helper function,
Split_DF_as_XY()

```python
def Split_DF_as_XY(df, train, dev, test):
    
    df = df.sample(frac=1).reset_index(drop=True)
    original_len = df.shape[0]
    i = int(train * original_len)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    
    train_df = df[:i].sample(frac=1).reset_index(drop=True)
    vectorizer.fit(train_df['comment'])
    
    x_train = vectorizer.transform(train_df['comment'])
    y_train = np.asarray(train_df['rating'])
    
    df = df[i:].sample(frac=1).reset_index(drop=True)
    i = int(dev * original_len)
    
    dev_df = df[:i].sample(frac=1).reset_index(drop=True)
    x_dev = vectorizer.transform(dev_df['comment'])
    y_dev = np.asarray(dev_df['rating'])
    
    df = df[i:].sample(frac=1).reset_index(drop=True)
    i = int(test * original_len)
        
    test_df = df[:i].sample(frac=1).reset_index(drop=True)
    x_test = vectorizer.transform(test_df['comment'])
    y_test = np.asarray(test_df['rating'])
    
    return [vectorizer, x_train, y_train, x_dev, y_dev, x_test, y_test]




data_sets = Split_DF_as_XY(scaled_df, train=.7, dev=.15, test=.15)
```
At long last, we now finally have our data in the appropriate format and we are
ready to begin experimentation.

Our first question is, What classifier maybe best suited for the given data?

Because the data is text, we will try our luck with Naive Bayes. However, becuase
we are going to test the data with Naive Bayes, we must create a helper fucntion
to convert Y values to integers so that Naive Bayes can correcly function.

So we create the helper function, Ratings_toInt() and also a dictionary accuracies
to record each classifier's accuracy with the data.

```python
accuracies = {}

def Ratings_toInt(ratings_float):
    
    ratings_int = ratings_float.astype(int)
    
    for i in range(ratings_float.shape[0]):
        
        distance = ratings_float[i] - float(ratings_int[i])
        if distance > .5: ratings_int[i] += 1
    
    return ratings_int
```
Ratings_toInt will convert the float rating to an integer rating by following
the rule,
        if rating - int(rating) > .5 then rating = rating + 1
        else rating is equal to rating

And now we create and test a Naive Bayes classifier.

```python
vectorizer, x_train, y_train, x_dev, y_dev, x_test, y_test = data_sets.copy()

y_train = Ratings_toInt(y_train)
y_dev = Ratings_toInt(y_dev)


nb = MultinomialNB(alpha=1)
nb.fit(x_train, y_train)

y_pred = nb.predict(x_dev)

accuracy = accuracy_score(y_dev, y_pred)*100
accuracies['NB'] = accuracy
print('Naive Bayes accuracy: {:.3f}%\n'.format(accuracy))
```

    Naive Bayes accuracy: 27.968%
    

Yikes, that is very low accuracy and thats with the hyperparameter alpha=1 for
Laplace smoothing and the ratings as integers and not even floats. Clearly this
job is too much for Naive Bayes. So which classifier should we try next?

Well, becuase we had to vectorize our x values, our next candidate can be KNN
becasue KNN can measure simularity of our x vectors. So now we ceate a KNN
classifier.

Again we must make use of our helper function Ratings_toInt as KNN requires
integer values to function correcly.

```python
vectorizer, x_train, y_train, x_dev, y_dev, x_test, y_test = data_sets.copy()

y_train = Ratings_toInt(y_train)
y_dev = Ratings_toInt(y_dev)


best_k = 0; highest_accuracy = 0
for k in [1, 2, 3, 5, 7, 9, 11, 17]:
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train) 

    y_pred = knn.predict(x_dev)

    accuracy = accuracy_score(y_dev, y_pred)*100
    if accuracy > highest_accuracy: best_k = k; highest_accuracy = accuracy
    print('KNN k = ' + str(k) + ', accuracy: {:.3f}%\n'.format(accuracy))
print()

accuracies['KNN'] = highest_accuracy
```

    KNN k = 1, accuracy: 28.574%
    
    KNN k = 2, accuracy: 27.715%
    
    KNN k = 3, accuracy: 27.676%
    
    KNN k = 5, accuracy: 22.556%
    
    KNN k = 7, accuracy: 23.423%
    
    KNN k = 9, accuracy: 23.768%
    
    KNN k = 11, accuracy: 24.122%
    
    KNN k = 17, accuracy: 24.431%
    
    

Wow, again, that is very low accuracy and thats with the hyperparameter
n_neighbors equal to k from k in [1, 2, 3, 5, 7, 9, 11, 17]. So clearly
this job is too much for KNN too. So which classifier should we try next?

Next we can try SVM or Support Vector Machine, because typically it is very
good for when we have no idea on the data, which seems to be the case given
that Naive Bayes and KNN have performed so terribly with the data at hand.

So now we define and test a SVM classifier. Yet again, we must make use of
our helper function Ratings_toInt as SVM requires integer values to function
correcly.

```python
vectorizer, x_train, y_train, x_dev, y_dev, x_test, y_test = data_sets.copy()

int_train = Ratings_toInt(y_train)
int_dev = Ratings_toInt(y_dev)


svm = LinearSVC()
svm.fit(x_train, int_train)

y_pred = svm.predict(x_dev)

accuracy = accuracy_score(int_dev, y_pred)*100
print('SVM accuracy: {:.3f}%\n'.format(accuracy))
accuracies['SVM'] = accuracy
```

    SVM accuracy: 32.444%
    

Ok, well given the accuracy we saw with Naive Bayes and KNN, this is actually
quite an improvement. Also, note that this accuracy was achieved without any
hyperparameter tuning.

Now let us review the accuracies that we have recorded for each classifier.

```python
figure = plt.figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k').add_subplot()        
figure.set_title('Classifiers & Their Accuracies')
figure.set_ylabel('Accuracy')
figure.set_xticklabels(['NB', 'KNN', 'SVM']) 

plt.bar('NB', accuracies['NB']/100)
plt.bar('KNN', accuracies['KNN']/100)
figure = plt.bar('SVM', accuracies['SVM']/100)
accuracies = []
```


{{< figure library="true" src="p3.jpg" lightbox="true" >}}

The bar graph above shows the comparison of each classifier and its accuracy against
that of the other classifiers and their accuracies. SVM's performance can be seen to
stand out here. However, this accuracy with SVM is still very low.

Our next question is, Can we improve this accuracy?

Let us create an experiment for SVM on a more simplified version of our problem.
We will create a model of SVM that can classify a comment as either positve or negative.

Below is the testing of this model.

```python
vectorizer, x_train, y_train, x_dev, y_dev, x_test, y_test = data_sets.copy()

int_train = Ratings_toInt(y_train)
b_train = np.asarray([1 if y >= 5 else 0 for y in int_train])

int_dev = Ratings_toInt(y_dev)
b_dev = np.asarray([1 if y >= 5 else 0 for y in int_dev])


svm = LinearSVC()
svm.fit(x_train, b_train)

y_pred = svm.predict(x_dev)

accuracy = accuracy_score(b_dev, y_pred)*100
accuracies.append(accuracy)
print('SVM: binary accuracy: {:.3f}%\n'.format(accuracy))
```

    SVM: binary accuracy: 91.479%
    

Wow, now that is impressive accuracy. However, the problem is quite simplified.
We need this kind of accuracy, but with multiclasses.

Let us perform some more experiments to see what we are dealing with. Can we
split the space into 3 with this kind of accuracy?

Below is the testing of this model.

```python
vectorizer, x_train, y_train, x_dev, y_dev, x_test, y_test = data_sets.copy()

int_train = Ratings_toInt(y_train)
tri_train = []
for y in int_train:
    if y < 4: tri_train.append(0)
    elif y < 8: tri_train.append(1)
    else: tri_train.append(2)
tri_train = np.asarray(tri_train)

int_dev = Ratings_toInt(y_dev)
tri_dev = []
for y in int_dev:
    if y < 4: tri_dev.append(0)
    elif y < 8: tri_dev.append(1)
    else: tri_dev.append(2)
tri_dev = np.asarray(tri_dev)


svm = LinearSVC()
svm.fit(x_train, tri_train)

y_pred = svm.predict(x_dev)

accuracy = accuracy_score(tri_dev, y_pred)*100
accuracies.append(accuracy)
print('SVM: trinary accuracy: {:.3f}%\n'.format(accuracy))
```

    SVM: trinary accuracy: 70.670%
    

Oh no, we lost a lot of our accuracy, but we still have a decent level. However,
taking the problem from binary to trinary may not be the best decision, but let
us see what happens if we ask SVM to seperate the space into 4.

Below is the testing of this new model.

```python
vectorizer, x_train, y_train, x_dev, y_dev, x_test, y_test = data_sets.copy()

int_train = Ratings_toInt(y_train)
quad_train = []
for y in int_train:
    if y < 4: quad_train.append(0)
    elif y < 6: quad_train.append(1)
    elif y < 8: quad_train.append(2)
    else: quad_train.append(3)
quad_train = np.asarray(quad_train)

int_dev = Ratings_toInt(y_dev)
quad_dev = []
for y in int_dev:
    if y < 4: quad_dev.append(0)
    elif y < 6: quad_dev.append(1)
    elif y < 8: quad_dev.append(2)
    else: quad_dev.append(3)
quad_dev = np.asarray(quad_dev)


svm = LinearSVC()
svm.fit(x_train, quad_train)

y_pred = svm.predict(x_dev)

accuracy = accuracy_score(quad_dev, y_pred)*100
accuracies.append(accuracy)
print('SVM: quadnary accuracy: {:.3f}%\n'.format(accuracy))
```

    SVM: quadnary accuracy: 58.371%
    

Oh no, that is an even bigger reduction in accuracy. Taking the problem further
from binary may not be the best decision.

Let's look at the changes in accuracy that occurred as we increasaed the the
number of spaces that we wanted to seperate the data into with SVM, in graph form.

```python
figure = plt.figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k').add_subplot()        
figure.set_xlabel('Sentiment Range Space Split Values')
figure.set_title('Sentiment Range Space Split Values vs Accuracy')
figure.set_ylabel('Accuracy')
figure.set_xticks(np.arange(0, 5, 1))
figure = figure.plot([2, 3, 4], [accuracy/100 for accuracy in accuracies], color='red', linewidth=2)
```


{{< figure library="true" src="p4.jpg" lightbox="true" >}}

From the graph above we can see that as we increase the number of classes,
we decrease SVM's accuracy. So that did not work. But we did get really good
accuracy when the problem was binary, so is there some way to keep the
problem binary and get higher accuracy than SVM on its own?

Next we will define a model that will use SVM classifiers to make binary
decisiones until an overall classification can be made. The model will have 1
classifier for each class. Each classifier will attempt to decide whether the
comment's rating is on the left or the right side of the rating continuum.
Once the algorithm detects that which side the rating is in, it will make its
prediction.

Below is the testing of this Major_Classifier, which when given a comment will
attempt to make a predciton as to what the rating is, to the nearest whole number.

```python
class Major_Classifier:
    
    def __init__(self, vectorizer, C=1):
        
        self.vectorizer = vectorizer
        self.C = C
        
        return
    
    
    
    def Train(self, x_train, y_train):
        
        self.labels = [label for label in set([y for y in y_train])]
        self.labels.sort()
        
        self.classifiers = {}
        for label in self.labels:
            
            if label != 10:
                Y = np.asarray([0 if y > label else label for y in y_train])
            else:
                Y = np.asarray([0 if y < label else label for y in y_train])
                    
            self.classifiers[label] = LinearSVC(C=self.C)
            self.classifiers[label].fit(x_train, Y)
        
        return
    
    
    
    def Predict(self, X):
        
        if isinstance(X, scipy.sparse.csr.csr_matrix) != True:
            if isinstance(X, list) == False: X = [X]
            V = self.vectorizer.transform(X)
        else: V = X
        
        predictions = []
        for v in V:
            p = 0
            for label in self.labels:
                p = self.classifiers[label].predict(v)[0]
                if p == label: break
            
            if p == 0: print('p == 0')
            
            predictions.append(p)
        
        predictions = np.asarray(predictions)
        
        return predictions



vectorizer, x_train, y_train, x_dev, y_dev, x_test, y_test = data_sets.copy()

int_train = Ratings_toInt(y_train)
int_dev = Ratings_toInt(y_dev)


major_classifier = Major_Classifier(vectorizer)
major_classifier.Train(x_train, int_train)

p = major_classifier.Predict(x_dev)

a = accuracy_score(int_dev, p)*100
print('major classifier accuracy: {:.3f}%\n'.format(a))
```

    major classifier accuracy: 35.160%
    

Thats better than what SVM can do all on its own! We did it, we created a model
that gets better accuracy than what SVM does on its own. Yes, this accuracy may
seem low but it is much better than randomly picking a value, where the
probability of randomly picking correctly is about 10%.

Now it is time to tune our hyperparameters. We must find a hyperparameter C, or
the Regularization parameter, that can possible achieve a higher accuracy.

Below we test different values for C, display a plotted relationship between
C values and accuracy, and then we select the best value for C.

```python
vectorizer, x_train, y_train, x_dev, y_dev, x_test, y_test = data_sets.copy()

int_train = Ratings_toInt(y_train)
int_dev = Ratings_toInt(y_dev)


accuracies = []; print()
for i, C in enumerate([.7, .5, .01, .001]):
    major_classifier = Major_Classifier(vectorizer, C)
    major_classifier.Train(x_train, int_train)

    p = major_classifier.Predict(x_dev)

    a = accuracy_score(int_dev, p)*100
    accuracies.append([C, a])
    print('major classifier, C = ' + str(C) + ', accuracy: {:.3f}%\n'.format(a))

C = []; A = []; max_a = 0; max_ai = 0; optimal_hyperparameter = 0
for i, (c, a) in enumerate(accuracies):
    C.append(c); A.append(a/100)
    if a > max_a: max_a = a; max_ai = i; optimal_hyperparameter = c

print('\noptimal value of C: ', C[max_ai])
print('          accuracy: {:.3f}%\n'.format(A[max_ai]*100))


figure = plt.figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k').add_subplot()        
figure.plot(C, A, color='blue', linewidth=2)
figure.set_xlabel('Regularization Parameter or C Values')
figure.set_title('Regularization Parameter (C Values) vs Accuracy')
figure = figure.set_ylabel('Accuracy')
```

    
    major classifier, C = 0.7, accuracy: 35.448%
    
    major classifier, C = 0.5, accuracy: 35.719%
    
    major classifier, C = 0.01, accuracy: 33.071%
    
    major classifier, C = 0.001, accuracy: 28.726%
    
    
    optimal value of C:  0.5
              accuracy: 35.719%
    



{{< figure library="true" src="p5.jpg" lightbox="true" >}}

From the above graph we can find our optimal hyperparameter for C. With this optimal
hyperparameter we will now test our model's final accuracy with test data set.

```python
vectorizer, x_train, y_train, x_dev, y_dev, x_test, y_test = data_sets.copy()

int_train = Ratings_toInt(y_train)
int_test = Ratings_toInt(y_test)


major_classifier = Major_Classifier(vectorizer, optimal_hyperparameter)
major_classifier.Train(x_train, int_train)

p = major_classifier.Predict(x_test)

a = accuracy_score(int_test, p)*100
print('major classifier final accuracy: {:.3f}%\n'.format(a))
```

    major classifier final accuracy: 35.654%
    

Nice, our final accuracy is a good improvement over what SVM could do when it
was only by itself. We created a model that outperformed SVM and then we tuned
this model's hyperparameters to find an optimal value. We then outperformed SVM
yet again.

Thank you for reading my blog post,

Again thank you,

Zacary Cotton
