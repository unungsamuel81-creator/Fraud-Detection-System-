import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from model import train_model,evaluate_model
from data_processing import load_and_preprocess_data

df =pd.read_csv("creditcard.csv")
print("First five transaction:")
print(df.head())

print("\nDataset dimensions:")
print(df.shape)

print("\nDataset column names:")
print(df.columns.tolist())

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nNumber of Duplicate row:")
print(df.duplicated().sum())

print("\nTransaction class distribution:")
print(df["Class"].value_counts())

class_percentages =df["Class"].value_counts(normalize=True)*100
print("\nTransaction Class percentages:")
print(class_percentages.round(2))

class_counts=df["Class"].value_counts().sort_index()
class_counts.plot(kind="bar")

plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Transactions Class(0 = Legitimate, 1=Fraudulent)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.show()

#Amount Vs class
sns.boxplot(x="Class", y="Amount", data=df)
plt.show()

print("\n-------STARTING MODEL TRAINING-----")
x_train, x_test, y_train, y_test =load_and_preprocess_data("creditcard.csv")
model = train_model(x_train, y_train)
evaluate_model(model, x_test, y_test)