from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    SERVICE_COLUMNS = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X["AverageMonthlySpend"] = np.where(
            X["tenure"] > 0,
            X["TotalCharges"] / X["tenure"],
            0,
        )
        X["ServiceCount"] = X[self.SERVICE_COLUMNS].eq("Yes").sum(axis=1)
        return X
