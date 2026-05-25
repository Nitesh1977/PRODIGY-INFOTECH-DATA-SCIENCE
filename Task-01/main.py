import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

# CHART 1: Age Distribution
def age():
    plt.figure(figsize=(8, 5))
    plt.hist(df['Age'].dropna(), bins=20, color='steelblue', edgecolor='black')
    plt.title('Distribution of Age - Titanic')
    plt.xlabel('Age')
    plt.ylabel('Number of Passengers')
    plt.tight_layout()
    plt.savefig('age_distribution.png')
    plt.show()

# CHART 2: Gender Distribution
def gender():
    plt.figure(figsize=(6, 4))
    df['Sex'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'], edgecolor='black')
    plt.title('Distribution of Gender - Titanic')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('gender_distribution.png')
    plt.show()

age()
gender()