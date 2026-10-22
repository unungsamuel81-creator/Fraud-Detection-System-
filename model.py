from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import pickle
import pandas as pd
import os

def train_model(x_train, y_train):
    #Smote to balance minority class
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    print(f"After SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(x_train_resampled, y_train_resampled)
    return model

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("ROC AUC: ", roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))

    #Save Model
    os.makedirs("models", exist_ok=True)
    with open('models/fraud_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Model saved to models/fraud_model.pkl")