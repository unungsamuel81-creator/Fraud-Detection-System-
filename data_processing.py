import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(path):
    df = pd.read_csv(path)

    #Scale Amount and Time
    scaler = StandardScaler()
    df["Amount"]= scaler.fit_transform(df[["Amount"]])

    #Feature and target
    x =df.drop("Class", axis=1)
    y =df["Class"]

    #Split --Use stratify because data is imbalanced
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)
    return x_train, x_test, y_train, y_test