import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load and preprocess the wine quality dataset
def BuildDataset():
    data = pd.read_csv("winequality-red.csv")
    # Binarize labels: good (quality >= 6) vs. bad (quality < 6)
    data['quality'] = np.where(data['quality'] >= 6, 1, 0)
    X = data.iloc[:, :-1].values
    y = data['quality'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test