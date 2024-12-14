import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import joblib

# Load the data
df = pd.read_excel("/content/yenideğişkenliproje (1).xlsx")

# Prepare the data
X = df.drop(columns=["Yolcu Sayısı", "Sefer Tarihi ve Saati"])
y = df["Yolcu Sayısı"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)

# Identify categorical and numerical columns
categorical = X.select_dtypes(include="object").columns
numeric = X.select_dtypes(exclude="object").columns
print("Categorical columns:", categorical)
print("Numerical columns:", numeric)

# Define preprocessing pipeline for categorical columns
categorical_pipeline = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

# Define preprocessing pipeline for numerical columns
numeric_pipeline = Pipeline(steps=[
    ("imputer", KNNImputer(n_neighbors=5)),
    ("scaler", RobustScaler())
])

# Combine both pipelines using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_pipeline, categorical),
    ("num", numeric_pipeline, numeric)
])

# Initialize the model (XGBoost Regressor)
model = XGBRegressor(objective='reg:squarederror', learning_rate=0.05, n_estimators=1000)

# Create a full pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_val_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_val_pred)
print(f"Mean Squared Error: {mse}")

# Save the entire pipeline (including the preprocessor and model)
joblib.dump(pipeline, "model_pipeline.joblib")

# Optionally, save only the model or preprocessor separately if needed
joblib.dump(model, "xgb_model.joblib")
joblib.dump(preprocessor, "preprocessing_pipeline.joblib")
