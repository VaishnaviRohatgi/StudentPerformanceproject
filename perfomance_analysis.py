import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix

sns.set_style('darkgrid')

df = pd.read_csv(r'C:\Users\Dell\Desktop\xAPI-Edu-Data.csv')

df.head()
print("the shape of our dataset is {}".format(df.shape))
df.info()
df.dropna()
df.isnull().sum()
df.nunique()

#Exploratory Data Analysis

df.drop('NationalITy', axis=1, inplace=True)

plt.figure(figsize=(14,8))
sns.countplot(x='PlaceofBirth', data=df)
plt.xlabel('nationality')
plt.ylabel('number of students')
plt.show()

sorted_place = df['PlaceofBirth'].value_counts().index

plt.figure(figsize=(14,8))
sns.countplot(x='PlaceofBirth', hue='gender', data=df, order=sorted_place)
plt.xlabel('nationality')
plt.ylabel('number of students')
plt.legend(loc='upper right')
plt.show()

fig, ax = plt.subplots(1,2, figsize=(14,8))
sns.countplot(x='StageID', data=df, ax=ax[0])
sns.countplot(x='StageID', hue='gender', data=df, ax=ax[1])
plt.show()


labels = df['Topic'].unique()

plt.figure(figsize=(14,8))
f = sns.countplot(x='Topic', data=df)
f.set_xticklabels(labels=labels, rotation=120)
plt.show()

fig, ax = plt.subplots(1,2, figsize=(14,8))
sns.countplot(x='Topic', hue='gender', data=df, ax=ax[0])
sns.countplot(x='Topic', hue='StageID', data=df, ax=ax[1])
ax[0].set_ylabel("number of students")
ax[1].set_ylabel("number of students")
ax[0].set_xticklabels(labels=labels, rotation=120)
ax[1].set_xticklabels(labels=labels, rotation=120)
ax[1].legend(loc='upper right')
plt.show()

topic_order = df['Topic'].value_counts().index

fig, ax = plt.subplots(2,2, figsize=(12,8))
sns.countplot(x=df.query('PlaceofBirth=="KuwaIT"')['Topic'], ax=ax[0,0], order=topic_order)
sns.countplot(x=df.query('PlaceofBirth=="Jordan"')['Topic'], ax=ax[0,1], order=topic_order)
sns.countplot(x=df.query('PlaceofBirth=="Iraq"')['Topic'], ax=ax[1,0], order=topic_order)
sns.countplot(x=df.query('PlaceofBirth=="USA"')['Topic'], ax=ax[1,1], order=topic_order)
ax[0,0].set_xticklabels(labels=labels, rotation=120)
ax[0,1].set_xticklabels(labels=labels, rotation=120)
ax[1,0].set_xticklabels(labels=labels, rotation=120)
ax[1,1].set_xticklabels(labels=labels, rotation=120)
ax[0,0].set_title('Kuwait')
ax[0,1].set_title('Jordan')
ax[1,0].set_title('Iraq')
ax[1,1].set_title('USA')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14,8))
sns.countplot(x='Class', data=df, ax=ax[0])
sns.countplot(x='Class', hue='StudentAbsenceDays', data=df, ax=ax[1])
plt.show()

fig, ax = plt.subplots(2, 2, figsize=(14,8))
sns.barplot(x='Class', y='raisedhands', data=df, ax=ax[0,0])
sns.barplot(x='Class', y='VisITedResources', data=df, ax=ax[0,1])
sns.barplot(x='Class', y='Discussion', data=df, ax=ax[1,0])
sns.barplot(x='Class', y='AnnouncementsView', data=df, ax=ax[1,1])
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,8))
sns.countplot(x='Class', hue='gender', data=df)
plt.show()

plt.figure(figsize=(12,8))
sns.countplot(x='gender', hue='StudentAbsenceDays', data=df)
plt.show()

sns.countplot(x='SectionID', hue='Class', data=df)
plt.show()

plt.figure(figsize=(14,8))
sns.countplot(x='Topic', hue='ParentschoolSatisfaction', data=df);
plt.show()

# Classification model

X = df.drop('Class', axis=1)

y = df['Class']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("the shape of the training set is {}".format(X_train.shape))
print("the shape of the test set is {}".format(X_test.shape))

# knn algorithm

classifier = KNeighborsClassifier(n_neighbors=12)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print('Accuracy score using KNN algorithm:', accuracy_score(y_pred, y_test))
print('confusion matrix\n', confusion_matrix(y_test, y_pred))
print('classification report\n', classification_report(y_test, y_pred))

# Logistic Regression

model = LogisticRegression(max_iter=10000)
model.fit(X_train , y_train)
pred = model.predict(X_test)

print('Accuracy score using Logistic Regression:', accuracy_score(pred, y_test))
print('confusion matrix\n', confusion_matrix(y_test, pred))
print('classification report\n', classification_report(y_test, pred))








