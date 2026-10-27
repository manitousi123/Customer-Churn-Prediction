import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

def preprocess_data(df, target_col='Churn', test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline for the Telco churn dataset.
    Returns: X_train_res, X_test, y_train_res, y_test, feature_names
    """
    # Work on a copy so the caller's DataFrame is never modified
    df = df.copy()

    # TotalCharges is stored as text with blank strings; convert to numeric.
    # The 11 blanks are customers with tenure = 0 (nothing billed yet), so 0 is
    # more accurate than the median and avoids using test-set information.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # customerID is a unique identifier with no predictive value.
    # Must reassign: drop() returns a new DataFrame and doesn't modify in place.
    df = df.drop(columns=['customerID'])

    # Derived feature: true average monthly charge.
    # replace(0, 1) avoids division by zero for tenure = 0 rows without
    # biasing the other rows (unlike tenure + 1).
    df['AvgMonthlyCharge'] = df['TotalCharges'] / df['tenure'].replace(0, 1)

    # Derived feature: 1 if the customer has OnlineSecurity or TechSupport
    df['HasPremiumSupport'] = (
        (df['OnlineSecurity'] == 'Yes') | (df['TechSupport'] == 'Yes')
    ).astype(int)

    # Encode the target as 0/1 (1 = churned)
    df[target_col] = (df[target_col] == 'Yes').astype(int)

    # One-hot encode all text columns. Including both 'object' and 'str' works
    # across pandas 2 and 3. drop_first=True avoids the dummy variable trap,
    # and dtype=int gives 0/1 integers instead of bools.
    cat_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split BEFORE resampling so synthetic rows never leak into the test set.
    # stratify=y keeps the churn ratio (~26.5%) the same in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Tell SMOTENC which columns are categorical (binary 0/1) so it copies
    # their values instead of interpolating them into values like 0.37.
    # This also catches HasPremiumSupport, which is int64 rather than a dummy.
    cat_indices = [i for i, c in enumerate(X_train.columns) if X_train[c].nunique() <= 2]

    # Oversample the minority class (churners) in the TRAINING set only.
    # The test set stays untouched so evaluation reflects real-world data.
    smote = SMOTENC(categorical_features=cat_indices, random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test, X.columns.tolist()