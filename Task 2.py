#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings(action='ignore')


# In[2]:


df = pd.read_csv('titanic.csv')
df.head(2)


# In[3]:


df.tail(2)


# In[5]:


def handle_non_numerical_data(df):
    
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        #print(column,df[column].dtype)
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            
            column_contents = df[column].values.tolist()
            #finding just the uniques
            unique_elements = set(column_contents)
            # great, found them. 
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    # creating dict that contains new
                    # id per unique string
                    text_digit_vals[unique] = x
                    x+=1
            # now we map the new "id" vlaue
            # to replace the string. 
            df[column] = list(map(convert_to_int,df[column]))

    return df


# In[6]:


y_target = df['Survived']
# Y_target is reshaped
x_train = df[['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare','Embarked', 'Ticket']]
x_train = handle_non_numerical_data(x_train)
x_train.head()


# In[7]:


fare = pd.DataFrame(x_train['Fare'])
# Normalizing
min_max_scaler = preprocessing.MinMaxScaler()
newfare = min_max_scaler.fit_transform(fare)
x_train['Fare'] = newfare
x_train


# In[8]:


x_train.isnull().sum()


# In[9]:


# Fill the NAN values with the median values in the datasets
x_train['Age'] = x_train['Age'].fillna(x_train['Age'].median())
print("The count of null values " , x_train['Age'].isnull().sum())
print(x_train.head())


# In[10]:


x_train['Sex'] = x_train['Sex'].replace('male', 0)
x_train['Sex'] = x_train['Sex'].replace('female', 1)
# print(type(x_train))
corr = x_train.corr()
corr.style.background_gradient()


# In[11]:


def plot_corr(df,size=10):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
# plot_corr(x_train)
x_train.corr()
corr.style.background_gradient()


# In[12]:


# Dividing the data into train and test data set
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_target, test_size = 0.4, random_state = 40)


# In[13]:


clf = RandomForestClassifier()
clf.fit(X_train, Y_train)


# In[14]:


print(clf.predict(X_test))
print("Accuracy: ",clf.score(X_test, Y_test))


# In[15]:


## Testing the model.
test_df = pd.read_csv('test.csv')
test_df.head(3)


# In[16]:


### Preprocessing on the test data
test_df = test_df[['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Ticket', 'Embarked']]
test_df = handle_non_numerical_data(test_df)

fare = pd.DataFrame(test_df['Fare'])
min_max_scaler = preprocessing.MinMaxScaler()
newfare = min_max_scaler.fit_transform(fare)
test_df['Fare'] = newfare
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Sex'] = test_df['Sex'].replace('male', 0)
test_df['Sex'] = test_df['Sex'].replace('female', 1)
print(test_df.head())


# In[17]:


#Pie charts-It is used to visualize the distribution of categorical data.
counts0=df['Survived'].value_counts()
plt.title('The pie chart of passenegers who survived or not',size=24)
print(counts0.plot(kind='pie',autopct=lambda x:f'{x:.0f}%',startangle=90))
plt.show()


# ### Observation
# 
# - 38% of pepole are survived and rest 62% died.

# In[20]:


import seaborn as sns
# Count of passengers in each class
class_counts = df['Pclass'].value_counts()

# Percentage of passengers in each class
class_percentages = (class_counts / len(df)) * 100

# Print count and percentage for each class
for class_num, count, percentage in zip(class_counts.index, class_counts, class_percentages):
    print(f"Class {class_num}: Count = {count}, Percentage = {percentage:.2f}%")

# Create separate count plots for each class
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', data=df)
plt.title('Passenger Class Count')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# ### Observation
# - Maximum people travelling through class 3
# - 491 people travelling through class 3

# In[21]:


df['Sex'] = df['Sex'].replace(['male', 'female'], [0, 1])

# Count of each gender
gender_counts = df['Sex'].value_counts()

# Percentage of each gender
gender_percentages = (gender_counts / len(df)) * 100

# Print count and percentage for each gender
for gender_num, count, percentage in zip(gender_counts.index, gender_counts, gender_percentages):
    gender = 'Male' if gender_num == 1 else 'Female'
    print(f"{gender}: Count = {count}, Percentage = {percentage:.2f}%")

# Create separate count plots for each gender
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', data=df)
plt.title('Gender Count')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks([0, 1], ['Male', 'Female'])  # Set proper labels for x-axis


# In[22]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True)


# ### Final Observations
# - Chances of female survival is more than male
# - Travelling in Pclass 3 is deadliest
# - Passenger onboarded from C is survived more
# - Passenger travelling with small families had a higher chance of surviving the accident in comparision to people with large       families & travelling alone
# - 891 passengers travelling in Titanic ship
# - 62% passengers died in Titanic Incident
# - Most of passengers (491) travelling through class 3
# - Out of total passengers travelling, most of passengers are male.
# - 64% are male passengers
# - Maximum passengers are travelling alone
# - Most of Passengers on boarded on ship from port Southampton
# - Most of passengers travelling in Titanic having age in the range of 20 to 40
# - Most of passengers died from class 3
# - More female survived in Titanic Incident
# - Most of younger Passengers are survived & older passengers are died in Titanic Incident.
