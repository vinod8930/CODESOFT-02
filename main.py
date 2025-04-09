import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

# Load CSV with correct path and encoding
df = pd.read_csv('C:/Users/vinod/Downloads/IMDb Movies India.csv', encoding='ISO-8859-1')

print("Original columns:", df.columns.tolist())

# Combine actor columns into a single "Actors" column
df['Actors'] = df['Actor 1'].fillna('') + ', ' + df['Actor 2'].fillna('') + ', ' + df['Actor 3'].fillna('')

# Filter only needed columns
df = df[['Genre', 'Director', 'Actors', 'Rating']]

# Drop missing ratings
df.dropna(subset=['Rating'], inplace=True)

# Fill missing categorical values
df.fillna({'Genre': 'Unknown', 'Director': 'Unknown', 'Actors': 'Unknown'}, inplace=True)

# Simplify to first genre/director/actor
df['Genre'] = df['Genre'].apply(lambda x: x.split(',')[0])
df['Director'] = df['Director'].apply(lambda x: x.split(',')[0])
df['Actors'] = df['Actors'].apply(lambda x: x.split(',')[0])

# Define features and target
X = df[['Genre', 'Director', 'Actors']]
y = df['Rating']

# One-hot encoding
categorical_features = ['Genre', 'Director', 'Actors']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nRoot Mean Squared Error (RMSE): {rmse:.2f}")
print("\nSample Predictions:")
print(pd.DataFrame({'Actual': y_test.values[:5], 'Predicted': y_pred[:5]}))
