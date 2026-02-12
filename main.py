# ==========================================
# MUSIC POPULARITY PREDICTION - FULL PROJECT
# ==========================================

# ------------------------------
# STEP 0: Import Libraries
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ------------------------------
# STEP 1: Load Dataset
# ------------------------------
df = pd.read_csv("songs.csv")

print("\nDataset Shape:", df.shape)
print("\nDataset Info:")
df.info()


# ------------------------------
# STEP 2: Handle Missing Values
# ------------------------------
df = df.dropna()


# ------------------------------
# STEP 3: Drop Irrelevant / Text Columns
# ------------------------------
df.drop(['artist_name', 'song_name', 'lyrics'], axis=1, inplace=True)


# ------------------------------
# STEP 4: Create Target Variable
# ------------------------------
median_popularity = df['new_artist_popularity'].median()

df['popular'] = df['new_artist_popularity'].apply(
    lambda x: 1 if x >= median_popularity else 0
)

df.drop('new_artist_popularity', axis=1, inplace=True)


# ------------------------------
# STEP 5: Encode Categorical Columns
# ------------------------------
label_encoder = LabelEncoder()

categorical_cols = ['language', 'genres']

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])


# ------------------------------
# STEP 6: Final Data Check
# ------------------------------
print("\nFinal Data Types:\n", df.dtypes)


# ------------------------------
# STEP 7: Split Features & Target
# ------------------------------
X = df.drop('popular', axis=1)
y = df['popular']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ------------------------------
# STEP 8: Feature Scaling
# ------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =====================================================
# BASELINE MODEL: LOGISTIC REGRESSION
# =====================================================
print("\n================ LOGISTIC REGRESSION ================")

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print("Classification Report:\n", classification_report(y_test, lr_pred))


# =====================================================
# IMPROVED MODEL: RANDOM FOREST
# =====================================================
print("\n================ RANDOM FOREST ================")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))


# ------------------------------
# STEP 9: Feature Importance
# ------------------------------
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features:\n")
print(feature_importance.head(10))


# ------------------------------
# STEP 10: Feature Importance Plot
# ------------------------------
plt.figure(figsize=(10, 6))
feature_importance.head(10).plot(kind='bar')
plt.title("Top 10 Features Affecting Song Popularity")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


# ------------------------------
# STEP 11: Simple EDA (Optional but Powerful)
# ------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(df['energy'], bins=30, kde=True)
plt.title("Energy Distribution")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(df['danceability'], bins=30, kde=True)
plt.title("Danceability Distribution")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(df['loudness'], bins=30, kde=True)
plt.title("Loudness Distribution")
plt.show()
