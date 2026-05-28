import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer



pd.set_option("display.max_columns",None)
pd.set_option('display.width', None)


#Reading Dataset
df=pd.read_csv("titanic.csv")
print(df.shape)
print(df.info())
print(df.isna().sum())
print(df["Sex"].value_counts())
print(df.head())
print(df["Embarked"].value_counts())

#Colormap to find the NAN values
plt.figure()
col=df.columns
color=["Yellow","Red"]
sns.heatmap(df[col].isnull(),cmap=sns.color_palette(color))
plt.title("Heat Map to show NAN values")
plt.show()

print(df["Survived"].value_counts())
print(df["Ticket"].nunique())
df=df.drop(columns=["PassengerId","Ticket","Name","Cabin"])
print(df.head())

#Encoding the "Sex" column
label_encoder=LabelEncoder()
df["Sex"]=label_encoder.fit_transform(df["Sex"])
print(df.head())

#Encoding the "Sex" column
df["Embarked"]=label_encoder.fit_transform(df["Embarked"])
print(df.head())

correlation=df.corr()
plt.figure()
sns.heatmap(correlation,vmax=0.8,annot=True,square=True)
plt.show()

#Converting into feature and Target
Y=df["Survived"]
df2=df.drop(columns="Survived")
X=df2
print(X)

#Imputation
mean=int(X["Age"].mean())
print(mean)
X["Age"].fillna(mean,inplace=True)
print("*"*80)
print(X)

mode=df["Embarked"].mode()
print(mode[0])
X["Embarked"].fillna(mode,inplace=True)

#Checking for null values
print(X.isna().sum())

#Splitting the data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=42,test_size=0.2)

#initializing the classifier
nb_model=GaussianNB()
nb_model.fit(X_train,Y_train)

#Predictions
y_pred=nb_model.predict(X_test)

dict_predict={"Y_Test":Y_test,
              "Y_Pred":y_pred}

df_prediction=pd.DataFrame(dict_predict)
print(df_prediction)

#Evaluation_Metrics
Accuracy_score=accuracy_score(Y_test,y_pred)
print(f"Accuracy of the trained model is:{Accuracy_score}")

confusion_Matrix=confusion_matrix(Y_test,y_pred)
plt.figure()
sns.heatmap(confusion_Matrix,annot=True)
plt.show()

classication_report1=classification_report(Y_test,y_pred)
print(classication_report1)