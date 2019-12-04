# ML based SEARCH ENGINE
Searching is a difficult task as it takes so much time to perform it.
If we have a large dataset then if we do one to one searching then it will take so much of user time.

## Dataset :
We have Stack Overflow Dataset from Kaggle link : https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/data

So Now we have a task :
1. User will enter a query related to code.
2. We have to process the query.
3. Return Results matching to our query.

## Limitation :
1. I have 8GB RAM and the dataset is of 7GB thus using it will be difficult.
So we are using SQLite to process info.
2. We have to reduce data, So I am taking only questions which are related to C#, C++, C, JAVA and iOS

## Workflow :
1. SearchEngine_Data.ipynb :
In this notebook we are getting our data and removing duplicates.
Then we move on to select tags which we want.
We used multiprocessing to do so as using 4 cores together increased the speed and did work of 2.5 hrs in 1 hr.
We saved the new processed dataframe in Sqlite Database.

2. PreProcessing.ipynb :
In this notebook we are PreProcessing the data in Title i.e our Questions.
We are removing any html tags and spaces and other junk or stopwords from it.

3. SearchEngine_Data.ipynb :
In this notebook we are creating system to access the queries, i.e starting step of building our Prediction system.
We First Vectorized the whole data and used Pairwise distance between the query and database but the Results were not upto the marks.
TFIDF performed better than BOW.

4. ClassificationMachineLearning.ipynb :
As in 3rd step we were not able to get good Results, So what we gonna do is to use some Classic Machine Learning.
So What I did is used this data to make a machine learning model.
The Title is a string values so we used TFIDFVectorizer ass tfidf performed better than bow in 3rd step.
Next step we divided the model into train, cv, test.
As we had such a sparse vector we had 2 choices LR or SVM.
We performed on both Unigram and Bigram but on bigram it was overfitting.
Then we finally used LR with Unigram as its performance was better.

Then After Predicting the Programming language of query then we add that in our query. Cause mostly when we search something on stackoverflow we often add tag with our question.

Then we repeated the steps we did in 3rd Step and our results were far better.

## Future :
1. We can use w2v and tfidf weighted w2v. As i was limited with resources and hence couldn't do it.
2. Making a flask api to make it presentable. As we also have body of questions and we are returning indexes from the search we can use that index to showcase them in presentable way.
