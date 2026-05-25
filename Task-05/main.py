import pandas as pd 
import  matplotlib.pyplot as plt 
import seaborn as sns 
# Data Cleaning
df = pd.read_csv("US_Accidents_March23.csv",nrows=50000)
print(df.head())
print(df.info())
print(df.isnull().sum())   
df= df.drop(['End_Lat','End_Lng','Wind_Chill(F)','Precipitation(in)'],axis=1)
df=df[['Weather_Condition','Start_Time','State','Visibility(mi)','Temperature(F)','Severity']]
df=df.dropna()

# Conversion 
df['Start_Time']=pd.to_datetime(df['Start_Time'])
df['Hour']=df['Start_Time'].dt.hour

# Accidents by Hour
plt.figure(figsize=(10,5))
sns.countplot(x='Hour', data=df)
plt.title("Accidents by Hour")
plt.savefig("Accidents by Hour")
plt.show()

# Weather Conditions
plt.figure(figsize=(12,6))
sns.countplot(
    y='Weather_Condition',
    data=df,
    order=df['Weather_Condition'].value_counts().head(10).index
)
plt.title("Top Weather Conditions for Accidents")
plt.savefig("Weather Conditions")
plt.show()

# Top States
plt.figure(figsize=(12,6))
sns.countplot(
    y='State',
    data=df,
    order=df['State'].value_counts().head(10).index
)
plt.title("Top States with Accidents")
plt.savefig("Top States")
plt.show()

# Severity
plt.figure(figsize=(8,5))
sns.countplot(x='Severity', data=df)
plt.title("Accident Severity Count")
plt.savefig("Severity")
plt.show()