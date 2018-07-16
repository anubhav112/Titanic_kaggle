import sys
import pandas as pd
from sklearn import preprocessing,model_selection,neighbors,svm
from sklearn.linear_model import LogisticRegression
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

df=pd.read_csv('D:\\anubhav\\Codes\\Titanic_train.csv')
test_df=pd.read_csv('D:\\anubhav\\Codes\\Titanic_test.csv')

df.drop(['PassengerId','Name','Ticket','Cabin'],1,inplace=True)
Id=np.array(test_df['PassengerId'])
test_df.drop(['PassengerId','Name','Ticket','Cabin'],1,inplace=True)

# fill the nan's with appropriate values
df['Age'].fillna(df['Age'].mean(),inplace=True)
test_df['Age'].fillna(df['Age'].mean(),inplace=True)

df['Embarked'].fillna('S',inplace=True)
test_df['Embarked'].fillna('S',inplace=True)

test_df['Fare'].fillna(df['Fare'].mean(),inplace=True)

df['Sex']=df['Sex'].replace({"male":100.0,
	"female":-100.0})
df['Embarked']=df['Embarked'].replace({"S":100.0,
	"C":200.0,
	"Q":300.0})
test_df['Sex']=test_df['Sex'].replace({"male":100.0,
	"female":-100.0})
test_df['Embarked']=test_df['Embarked'].replace({"S":100.0,
	"C":200.0,
	"Q":300.0})

def handle_non_numeric_values(column_name):
	val=df[column_name].values.tolist()
	val=val+(test_df[column_name].values.tolist())
	unique_val=set(val)
	temp={}
	def just(val):
		return temp[val]
	k=0
	for i in unique_val:
		temp[i]=k
		k+=1
	df[column_name]=list(map(just,df[column_name]))
	test_df[column_name]=list(map(just,test_df[column_name]))
	return df


# df['Cabin'].fillna(Counter(np.array(df['Cabin'])).most_common(2)[1][0],inplace=True)
# test_df['Cabin'].fillna(Counter(np.array(df['Cabin'])).most_common(2)[1][0],inplace=True)
# handle_non_numeric_values('Cabin')

# Was trying plot Age and Survived to know which age group survived most
# X=df['Age']
# y=df['Survived']
# arr=[]
# for i in range(891):
#     if(df['Survived'][i]==1):
#         arr.append(df['Age'][i])
        
# plt.subplot(1,2,1)
# k,_,_=plt.hist(X,bins=30)
# k1,_,_=plt.hist(arr,bins=30)
# plt.show()
# arr1=[k1[i]/k[i] if(k[i] != 0) else 0 for i in range(len(k))]
# for i in range(len(arr1)):
#     print(i,arr1[i])

X_train=np.array(df.drop(['Survived'],1))
y_train=np.array(df[['Survived']])
X_test=np.array(test_df)
print(df.head())

X_train_set,X_test_set,y_train_set,y_test_set=model_selection.train_test_split(X_train,y_train,test_size=0.2)
clf=svm.SVC(C=5,kernel='linear')
clf.fit(X_train_set,y_train_set)
# y_test=clf.predict(X_test)

print(clf.score(X_test_set,y_test_set))
# sys.stdout=open("D:\\anubhav\\Codes\\Titanic.csv","w")
# print("PassengerId,Survived")
# for i in range(0,len(y_test)):
# 	print(str(Id[i])+","+str(y_test[i]))