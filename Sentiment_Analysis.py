# Natural Language Processing

# Importing the libraries

import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# delimiter to tell its tsv 
# quoting 3 to avoid quotes in data

# Cleaning the texts
import re    #pattern lib
import nltk    # to import list of irrelevant word + imp for nlp
nltk.download('stopwords')  #list of irrelevant word in many lang
from nltk.corpus import stopwords    #inpoting list
from nltk.stem.porter import PorterStemmer     #for stemming  (like==likes==liking)

corpus = []
for i in range(0, dataset.shape[0]): #cleaning one review at a time
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])   # to retian only a-zA-z char in review 
    review = review.lower()   # tolower
    review = review.split()   # string to list
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    #set used for faster implementation  
    #ps.stem(word) for stemming  
    review = ' '.join(review)
    #list of words to string
    corpus.append(review)

# Creating the Bag of Words model
 #its a model  with review vs distinc words matrix where each word act as a feature 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)  #maxfeatures takes words acc to their freq
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Efficiency (Naive Bayes)= "+ str((cm[0][0]+cm[1][1])/(sum(sum(cm)))) )



#Fitting Random Forest classifier model to training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10 , criterion='entropy' ,random_state=0 )
classifier.fit(X_train,y_train)


#Predicting test set results
y_pred = classifier.predict(X_test)


#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)

print("Efficiency (Random Forest)= "+ str((cm[0][0]+cm[1][1])/(sum(sum(cm)))) )

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Efficiency (SVM)= "+ str((cm[0][0]+cm[1][1])/(sum(sum(cm)))) )
