from os import PathLike
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import pathlib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


Data = pd.read_csv('data/data.csv')
Data.head()
print(Data.head())

x = Data.drop(columns='Bankrupt?',axis=1)
y = Data['Bankrupt?']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=42)
Model = RandomForestClassifier()
Model.fit(x_train,y_train)

dump(Model,pathlib.Path('model/model-v1.joblib'))
y_pred = Model.predict(x_test)


# evaluate the model with accuracy score
accuracy = accuracy_score(y_test,y_pred)
print(f"""The Accuracy : {accuracy * 100:.2f}%""")


# evaluate the model with classification report
print(classification_report(y_test,y_pred))


print(f"""Train Score : {Model.score(x_train,y_train) * 100:.2f}%""")
print(f"""Test Score : {Model.score(x_test,y_test) * 100:.2f}%""")
# display training and test accuracy

