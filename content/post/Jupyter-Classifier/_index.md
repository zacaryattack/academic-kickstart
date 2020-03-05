---
title: Practice Classifier in Jupyter
date: 2020-03-03

# Put any other Academic metadata here...

---


```python
from sys import path

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing



def p(item_list=None, snlc=0, title=None):

    if snlc > 0:
        print('\n'*snlc, end='')

    if title != None:
        print(title, end='')

    item_list = [item_list] if type(item_list) is not list else item_list
    for item in item_list:
        print('\n', item, '\n')

    return




NLP_Tutorial_path = path[0]

train_data_path = NLP_Tutorial_path + '/train.csv'
train_df = pd.read_csv(train_data_path)

test_data_path = NLP_Tutorial_path + '/test.csv'
test_df = pd.read_csv(test_data_path)




FAKE_disaster_tweet = train_df[train_df['target'] == 0]['text'].values[1]
#p(FAKE_disaster_tweet, title='FAKE_disaster_tweet')

REAL_disaster_tweet = train_df[train_df['target'] == 1]['text'].values[1]
#p(REAL_disaster_tweet, title='REAL_disaster_tweet')




# Using scikit-learn's CountVectorizer to count the words in each tweet
# and turn them into data the machine learning model can process.
count_vectorizer = feature_extraction.text.CountVectorizer()
#p(count_vectorizer, title='count_vectorizer')



# let's get counts for the first 5 tweets in the data
example_train_vectors = count_vectorizer.fit_transform(train_df['text'][0:5])
#p(example_train_vectors, snlc=3, title='example_train_vectors')



# we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)
result = example_train_vectors[0].todense().shape
#p(result, snlc=3, title='example_train_vectors[0].todense().shape')
result = example_train_vectors[0].todense()
#p(result, title='example_train_vectors[0].todense()')




# vectors for all of our tweets
train_vectors = count_vectorizer.fit_transform(train_df['text'])
#p(train_vectors, snlc=5)




# Note that we're NOT using .fit_transform() here.
# Using just .transform() makes sure that the tokens in the train vectors are the only ones 
# mapped to the test vectors - i.e. that the train and test vectors use the same set of tokens.
test_vectors = count_vectorizer.transform(test_df['text'])




# Becasue our vectors are really big, we want to push our model's weights
# toward 0 without completely discounting different words
# - ridge regression is a good way to do this.
Classifier = linear_model.RidgeClassifier()




#scores
scores = model_selection.cross_val_score(Classifier, train_vectors, train_df['target'], cv=3, scoring='f1')
#p(scores)




Classifier.fit(train_vectors, train_df['target'])




sample_submission_path = NLP_Tutorial_path + '/sample_submission.csv'
sample_submission = pd.read_csv(sample_submission_path)
#p(sample_submission)

sample_submission['target'] = Classifier.predict(test_vectors)
#p(sample_submission['target'])

sample_submission.head()

submission_path = NLP_Tutorial_path + '/submission.csv'
sample_submission.to_csv(submission_path, index=False)


print(sample_submission)

```

             id  target
    0         0       0
    1         2       1
    2         3       1
    3         9       0
    4        11       1
    ...     ...     ...
    3258  10861       1
    3259  10865       1
    3260  10868       1
    3261  10874       1
    3262  10875       0
    
    [3263 rows x 2 columns]



```python
Please view the .77096 score and place 2714 I received for my code in the picture below.
```

{{< figure library="true" src="ranking.jpg" title="A caption" lightbox="true" >}}
