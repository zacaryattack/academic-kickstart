---
title: Assignment 3
date: 2020-21-04

# Put any other Academic metadata here...
summary: "The goal of this assignment is to learn about the Naive Bayes Classifier (NBC)."
---

{{% staticref "files/Assignment 3.ipynb" "newtab" %}}Please download my ipynb code here.{{% /staticref %}}
 
To successfully execute this .ipynb file, the text dataset dirrectory aclImdb,
must be located within the same dirrectory as this .ipynb file.

The text dataset dirrectory aclImdb is obtained from
url: http://ai.stanford.edu/~amaas/data/sentiment/

We begin by importing the necessary libraries and building the necessary paths
in order to load the data sets, which upon download are located in a dir called
aclImdb, in a train dir and a test dir.

```python
import re # for formatting reviews

from math import log # for probability representation
from sys import path # to locate review txt files
from os import listdir # to locate review txt files
from os.path import join # to locate review txt files
from random import shuffle # to mix up lists
from collections import defaultdict # for holding results

from sklearn.metrics import accuracy_score # for producing accuracy
from sklearn.model_selection import train_test_split # for splitting data




# ensure that the .ipynb file is in the
# same dir as the unzipped aclImdb dir
data_dir_path = path[0] + '/aclImdb'

print()

train_dir_path = data_dir_path + '/train'
print(train_dir_path)
train_pos_path = train_dir_path + '/pos'
print(train_pos_path)
train_neg_path = train_dir_path + '/neg'
print(train_neg_path)

print()

test_dir_path = data_dir_path + '/test'
print(test_dir_path)
test_pos_path = test_dir_path + '/pos'
print(test_pos_path)
test_neg_path = test_dir_path + '/neg'
print(test_neg_path, end='\n\n')
```

    
    /Users/zmo/here/jupyter/data mining/ass3/aclImdb/train
    /Users/zmo/here/jupyter/data mining/ass3/aclImdb/train/pos
    /Users/zmo/here/jupyter/data mining/ass3/aclImdb/train/neg
    
    /Users/zmo/here/jupyter/data mining/ass3/aclImdb/test
    /Users/zmo/here/jupyter/data mining/ass3/aclImdb/test/pos
    /Users/zmo/here/jupyter/data mining/ass3/aclImdb/test/neg
    


After having built our paths, next we must read each document 1 by 1 and store
each's content in the form of a list, after that we must format each list to
rid any elements that are non alphanumeric or tailor excessive whitespaces and
set all alpha characters to lowercase.

```python
# returns list of reviews
def Load_Reviews(review_dir_path):
    
    # obatins a list of file paths from the supplied review_dir_path
    file_paths = [join(review_dir_path, file) for file in listdir(review_dir_path) if '.txt' in file]
    
    # each element of reviews is a read txt doc
    reviews = []
    for file_path in file_paths:
        with open(file_path, 'r') as file: reviews.append(file.read())

    return reviews




# does (s)light formatting of reviews
def Format_Reviews(reviews):
    
    # removes excessive whitespaces & converts to lowercase
    for i, review in enumerate(reviews):
        
        # only keep alphanumeric characters
        review = re.sub('[^0-9a-zA-Z]+', ' ', review.lower())
        reviews[i] = ' '.join(review.split())
    
    return reviews




# each element of train_pos is a read txt doc
train_pos = Load_Reviews(train_pos_path)
train_pos = Format_Reviews(train_pos)

# each element of train_neg is a read txt doc
train_neg = Load_Reviews(train_neg_path)
train_neg = Format_Reviews(train_neg)


# each element of test_pos is a read txt doc
test_pos = Load_Reviews(test_pos_path)
test_pos = Format_Reviews(test_pos)

# each element of test_neg is a read txt doc
test_neg = Load_Reviews(test_neg_path)
test_neg = Format_Reviews(test_neg)
```

a. Divide the data set as train, development, and test.

Now we will start part a. and combine the training and testing data that was
supplied in the movie review folders into a single pool of data, shuffle it,
and then split this pool into a train, development, and test set.

```python
x_training = train_pos + train_neg
y_training = [1]*len(train_pos) + [0]*len(train_neg)

x_testing = test_pos + test_neg
y_testing = [1]*len(test_pos) + [0]*len(test_neg)

x_pool = x_training + x_testing
y_pool = y_training + y_testing

# pools all data into a single source, shuffles it, then
# splits into x_train, y_train, x_dev, y_dev, x_test, y_test
# x_train, y_train get 70%
#     x_dev, y_dev get 15%
#   x_test, y_test get 15%
pool = list(zip(x_pool, y_pool))
shuffle(pool)

x_pool, y_pool = zip(*pool)
x_all, y_all = x_pool, y_pool

##############################################################
# a. Divide the data set as train, development, and test.
##############################################################
x_train, x_dev, y_train, y_dev = train_test_split(x_pool, y_pool, test_size=0.3, random_state=42)
x_dev, x_test, y_dev, y_test = train_test_split(x_dev, y_dev, test_size=0.5, random_state=42)
```

b. Build a vocabulary as set not list (for speed increase).

Now we define our Naive Bayes Classifier, it has 2 main central functions, Train
and Predict. After training the model on the train set we must then demonstrate
the accuracy of the model with the development set.

```python
class NaiveBayes_Classifier:
    
    def Train(self, x_train, y_train, smoothing_flag):
        
        # records the known class labels
        self.class_labels = set(y_train)
        
        # separates reviews by classes
        self.reviews_by_class = {}
        for i, review in enumerate(x_train):
            if y_train[i] in self.reviews_by_class: self.reviews_by_class[y_train[i]].append(review)
            else: self.reviews_by_class[y_train[i]] = [review]
        
        # locates length of class with smallest review count
        min_len = float('inf')
        for class_label in self.class_labels:
            shuffle(self.reviews_by_class[class_label])
            if len(self.reviews_by_class[class_label]) < min_len:
                min_len = len(self.reviews_by_class[class_label])
        
        # makes all classes have the same review count
        for class_label in self.class_labels:
            shuffle(self.reviews_by_class[class_label])
            self.reviews_by_class[class_label] = self.reviews_by_class[class_label][:min_len]
        
        
        ##############################################################
        # b. Build a vocabulary as set not list (for speed increase).
        ##############################################################
        # records the vocabulary and word stats for words by classes
        # builds a vocabulary as set becasue as a list was too slow!!
        self.vocab = set(); word_stats = {}
        for label in self.class_labels:
            word_stats[label] = {}

            for review in self.reviews_by_class[label]:
                for word in review.split():
                    
                    self.vocab.add(word)
                    if word in word_stats[label]: word_stats[label][word] += 1
                    else: word_stats[label][word] = 1
        
        # build priors & likelihoods
        # prior & likelihood but, use log to stop computer
        # from rounding to 0 so, log(prior) & log(likelihood)
        self.log_priors = {}; self.log_likelihoods = {}
        for label in self.class_labels:
            # builds a log(prior)
            self.log_priors[label] = log(len(self.reviews_by_class[label]) / len(x_train))
            
            # build a log(likelihood)
            self.log_likelihoods[label] = {}
            total_count_of_word = sum([word_stats[label][word] for word in self.vocab if word in word_stats[label]])
            for word in self.vocab:
                numerator = word_stats[label][word] if word in word_stats[label] else 0   
                denominator =  total_count_of_word
                
                # if smoothing, applies Laplace Smoothing
                if smoothing_flag == True: numerator += 1; denominator += len(self.vocab)
                
                if(
                    numerator == 0 or
                    denominator == 0
                ):
                    self.log_likelihoods[label][word] = 0
                
                else: self.log_likelihoods[label][word] = log(numerator/denominator)

        return
    
    
    
    def Get_WordsProbability(self, word):
        
        # num of documents containing ‘the’ / num of all documents
        
        word_in_review_count = 0
        total_review_count = 0
        
        for class_label in self.reviews_by_class:
            total_review_count += len(self.reviews_by_class[class_label])            
            for review in self.reviews_by_class[class_label]:
                if word in review: word_in_review_count += 1
                
        probability = word_in_review_count / total_review_count
        probability *= 100
                
        return probability
    
    
    
    def Get_ConditionalProb_ofWord_based_onSentiment(self, word, sentiment):
                
        # of positive documents containing “the” / num of all positive review documents
        
        word_in_pos_review_count = 0
        for review in self.reviews_by_class[sentiment]:
            if word in review: word_in_pos_review_count += 1
            
        total_pos_review_count = len(self.reviews_by_class[sentiment])
                
        probability = word_in_pos_review_count / total_pos_review_count
        probability *= 100
        
        return probability
    
    
    
    def Predict(self, reviews):
        
        # so reviews could be a single string or a list of strings
        if isinstance(reviews, list) != True: reviews = [reviews]
        
        predictions = [0]*len(reviews)
        for i, review in enumerate(reviews):
            
            probs = {}
            for label in self.class_labels:
                
                # for each class label builds the probability,
                # incrementally adds log_likelihoods to log_priors
                # for each given word in a given review
                probs[label] = self.log_priors[label]
                for word in review.split():
                    if word in self.vocab: probs[label] += self.log_likelihoods[label][word]
        
            # determines the prediction, picks class with highest probability
            if probs[0] < probs[1]: predictions[i] = 1
                
        return predictions




nbc = NaiveBayes_Classifier()
nbc.Train(x_train, y_train, smoothing_flag=True)

##############################################################
# c. Calculate the following probability
##############################################################
print('\nc. Calculate the following probability')

print('\nProbability of the occurrence:\n' + ' '*12 + 'P[\"the\"] = ?')
the_prob = nbc.Get_WordsProbability('the')
print(' '*12 + 'P[\"the\"] = {:.3f}%'.format(the_prob))

print('\nConditional Prob based on sentiment:\n P[\"the\" | Positive] = ?')
the_condprob = nbc.Get_ConditionalProb_ofWord_based_onSentiment('the', 1)
print(' P[\"the\" | Positive] = {:.3f}%\n'.format(the_condprob))
```
c. Calculate the following probability

    Probability of the occurrence:
                P["the"] = ?
                P["the"] = 99.543%
    
    Conditional Prob based on sentiment:
     P["the" | Positive] = ?
     P["the" | Positive] = 99.483%
    


d. Calculate accuracy using dev data set

Next we will perform part d., calculate accuracy using dev data set and use
k-fold cross validation to estimate the skill of the model with k = 5

```python
class K_Fold_CrossValidation:
    
    def Prepare_Evaluation(self, k, X, Y):
        
        self.k = k
        
        # shuffle the data set randomly
        data_set = list(zip(X, Y))        
        shuffle(data_set)
        
        data_set_len = len(data_set)
        while data_set_len % self.k != 0: data_set_len -= 1
        step = int(data_set_len / self.k)
        
        # split the data set into k groups
        self.groups = []; i = 0; j = step
        for k in range(self.k): self.groups.append(data_set[i:j]); i += step; j += step
        shuffle(self.groups)
                
        return
    
    
    
    def Evaluate(self):
                
        scores = []
        for i, group in enumerate(self.groups):
            # for each group, take the group as a development data set
            
            group = self.groups.pop(i)
            x_dev, y_dev = zip(*group); x_dev = list(x_dev); y_dev = list(y_dev)
            
            # take the remaining groups as a training data set
            train_group = []
            for group in self.groups: train_group += group
            x_train, y_train = zip(*train_group); x_train = list(x_train); y_train = list(y_train)
            
            # train a model on the training set & evaluate it on the development set
            nbc = NaiveBayes_Classifier()
            nbc.Train(x_train, y_train, smoothing_flag=True)

            y_pred = nbc.Predict(x_dev)
            accuracy = accuracy_score(y_dev, y_pred)*100
            
            # retain the accuracy score
            scores.append(accuracy)
            self.groups.insert(i, group)
        
        print('\nCalculated accuracies during conduction of five fold cross validation.')
        
        for i, score in enumerate(scores): print('   accuracy {}: {:.3f}%'.format(i+1, score))
        
        print()
        
        return




nbc = NaiveBayes_Classifier()
nbc.Train(x_train, y_train, smoothing_flag=True)

print('\nd. Calculate accuracy using dev data set')

y_pred = nbc.Predict(x_dev)
accuracy = accuracy_score(y_dev, y_pred)*100
print('\nCalculated accuracy using dev data set: {:.3f}%'.format(accuracy))

kfcv = K_Fold_CrossValidation()
kfcv.Prepare_Evaluation(5, x_all, y_all)
kfcv.Evaluate()
```

    
    d. Calculate accuracy using dev data set
    
    Calculated accuracy using dev data set: 84.800%
    
    Calculated accuracies during conduction of five fold cross validation.
       accuracy 1: 84.720%
       accuracy 2: 84.720%
       accuracy 3: 84.240%
       accuracy 4: 83.320%
       accuracy 5: 94.150%
    


e. Do following experiments (1)

Experiment 1: Compare the effects of smoothing.

```python
print('\ne. Doing following experiment (1): Compare the effects of Smoothing')

nbc = NaiveBayes_Classifier()

nbc.Train(x_train, y_train, smoothing_flag=False)
y_pred = nbc.Predict(x_dev)
accuracy_sf = accuracy_score(y_dev, y_pred)*100
print('\nNaive Bayes Classifier without Smoothing.')
print('Calculated accuracy using dev data set: {:.3f}%'.format(accuracy_sf))

nbc.Train(x_train, y_train, smoothing_flag=True)
y_pred = nbc.Predict(x_dev)
accuracy_st = accuracy_score(y_dev, y_pred)*100
print('\nNaive Bayes Classifier with Smoothing.')
print('Calculated accuracy using dev data set: {:.3f}%'.format(accuracy_st))

print('\nUsing smoothing resulted in accuracy that was {:.3f}% higher.\n'.format(accuracy_st-accuracy_sf))
```

    
    e. Doing following experiment (1): Compare the effects of Smoothing
    
    Naive Bayes Classifier without Smoothing.
    Calculated accuracy using dev data set: 67.080%
    
    Naive Bayes Classifier with Smoothing.
    Calculated accuracy using dev data set: 84.827%
    
    Using smoothing resulted in accuracy that was 17.747% higher.
    


e. Do following experiments (2)

Experiment 2: Derive top 10 words that predict positive and negative classes.

```python
print('\ne. Doing following experiment (2): Derive Top 10 words that predicts positive and negative classes')

results = defaultdict(list)
for class_label in nbc.class_labels:
    words = list(nbc.log_likelihoods[class_label].keys())
    
    # creates a list of words and their associated probabilities
    results[class_label] = [[word, nbc.log_likelihoods[class_label][word]] for word in words]
    
    # sorts list in descending order
    results[class_label].sort(reverse=True, key=lambda x: x[1])

words, probs = zip(*(results[1][:10]))
print('\ntop 10 words that predict the positive class')
for i, word in enumerate(words): print('  word ' + str(i+1) + ': ' + word)

words, probs = zip(*(results[0][:10]))
print('\ntop 10 words that predict the negative class')
for i, word in enumerate(words): print('  word ' + str(i+1) + ': ' + word)
print()
```

    
    e. Doing following experiment (2): Derive Top 10 words that predicts positive and negative classes
    
    top 10 words that predict the positive class
      word 1: the
      word 2: and
      word 3: a
      word 4: of
      word 5: to
      word 6: is
      word 7: in
      word 8: br
      word 9: it
      word 10: i
    
    top 10 words that predict the negative class
      word 1: the
      word 2: a
      word 3: and
      word 4: of
      word 5: to
      word 6: br
      word 7: is
      word 8: it
      word 9: i
      word 10: in
    


f. Using the test data set

Finally we will perform part f. We have seen that the effects of using
smoothing are much too impressive to neglect and so we train the model
again with the train set and with the optimal hyperparameter being to
use smoothing, we then calculate the FINAL accuracy with the test set.

```python
print('\nf. Using the test data set')

nbc = NaiveBayes_Classifier()
nbc.Train(x_train, y_train, smoothing_flag=True)

y_pred = nbc.Predict(x_test)

accuracy = accuracy_score(y_test, y_pred)*100
print('\nNaive Bayes Classifier with optimal hyperparameter: Smoothing')
print('\nCalculated FINAL accuracy using test data set: {:.3f}%\n'.format(accuracy_st))
```

    
    f. Using the test data set
    
    Naive Bayes Classifier with optimal hyperparameter: Smoothing
    
    Calculated FINAL accuracy using test data set: 84.827%
    

