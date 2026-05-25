import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('titanic.csv')

# ===== DATA CLEANING =====
# Fill Age with MEAN (SAHI TAREEKA)
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill Embarked with MODE
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin (too many missing)
df = df.drop('Cabin', axis=1)

# ===== FEATURE ENGINEERING =====
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# ===== VISUALIZATIONS =====
# 1. Survival by Gender
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Sex', hue='Survived', palette='Set1')
plt.title('Survival Count by Gender')
plt.savefig('gender.png')
plt.show()

# 2. Survival by Class
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Pclass', hue='Survived', palette='viridis')
plt.title('Survival Count by Passenger Class')
plt.savefig('class.png')
plt.show()

# 3. Age Distribution
plt.figure(figsize=(10, 5))
plt.hist(df['Age'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution of Passengers')
plt.savefig('age.png')
plt.show()