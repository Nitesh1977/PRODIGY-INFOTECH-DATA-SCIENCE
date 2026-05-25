import pandas as pd

# LOAD DATASET and cleaing 
df = pd.read_csv("bank.csv",sep=";")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df['y'].value_counts())
print(df.select_dtypes(include='object').columns)

# ENCODE CATEGORICAL VARIABLES
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col]=le.fit_transform(df[col])
print(df.head())
# FEATURES & TARGET SPLIT  
x=df.drop('y',axis=1)
y=df['y']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    test_size=0.2,
    random_state=42
)

# BUILD DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
model =DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred =model.predict(x_test)
print(y_pred)

# MAKING PREDICTIONS
from sklearn.metrics import accuracy_score 
print(accuracy_score(y_test,y_pred))