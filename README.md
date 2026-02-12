
# 🎵 Music Popularity Prediction using Machine Learning

## 📌 Project Overview

This project is an **end-to-end Data Science & Machine Learning application** that predicts whether a song is likely to be **popular or not** based on its audio and metadata features.

The goal of this project is to demonstrate the **complete data science workflow**, starting from raw data and ending with model evaluation and insights.
This project is designed to be **beginner-friendly** and suitable for **Data Analyst / Data Scientist / Business Analyst freshers**.

---

## 🎯 Problem Statement

Music streaming platforms release thousands of new songs every day.
It is not possible to promote every song equally.

**Problem:**
Can we predict whether a song will be popular based on its audio features?

**Solution:**
Build a machine learning model that classifies songs into:

* **Popular**
* **Not Popular**

---

## 📂 Dataset Description

* Dataset contains **41,574 songs**
* Each row represents one song
* Data includes audio characteristics and metadata

### Key Columns Used:

* `acousticness`
* `danceability`
* `energy`
* `loudness`
* `tempo`
* `valence`
* `duration_ms`
* `language`
* `genres`
* `new_artist_popularity` (used to create target)

### Columns Removed:

* `artist_name` → identifier only
* `song_name` → identifier only
* `lyrics` → text data (requires NLP, not used in this beginner project)

---

## 🧠 Machine Learning Task

This is a **Binary Classification Problem**.

### Target Variable Creation:

* The median value of `new_artist_popularity` is calculated
* Songs with popularity **above or equal to median** → `Popular (1)`
* Songs with popularity **below median** → `Not Popular (0)`

---

## 🛠️ Tools & Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* VS Code

---

## 🔄 Project Workflow (Step-by-Step)

### 1️⃣ Data Loading

* Loaded dataset using Pandas
* Checked shape, column names, and data types

### 2️⃣ Data Cleaning

* Removed missing values
* Dropped non-useful and text-based columns

### 3️⃣ Feature Engineering

* Created binary target variable (`popular`)
* Encoded categorical columns (`language`, `genres`)

### 4️⃣ Data Splitting

* Split data into training and testing sets (80% / 20%)

### 5️⃣ Feature Scaling

* Applied `StandardScaler` to normalize numerical features

---

## 🤖 Models Used

### 🔹 Logistic Regression (Baseline Model)

* Simple and interpretable model
* Used as a baseline for comparison
* Achieved ~63% accuracy

### 🔹 Random Forest Classifier (Improved Model)

* Ensemble learning model
* Captures non-linear relationships
* Improved accuracy to ~70–78%
* Provided feature importance insights

---

## 📊 Model Evaluation Metrics

* Accuracy Score
* Confusion Matrix
* Precision, Recall, F1-score

These metrics help evaluate how well the model performs on unseen data.

---

## 📈 Feature Importance (Business Insight)

Using Random Forest, the most important features influencing song popularity were:

* Energy
* Danceability
* Loudness
* Valence
* Tempo

**Insight:**
High-energy, danceable, and louder songs have a higher chance of becoming popular.

---

## 📉 Exploratory Data Analysis (EDA)

Basic visualizations were created to understand feature distributions:

* Energy distribution
* Danceability distribution
* Loudness distribution

EDA helped in understanding patterns before and after modeling.

---

## 🚀 How to Run This Project

### Step 1: Clone or Download Repository

```bash
git clone <repository-url>
```

### Step 2: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Step 3: Run the Project

```bash
python main.py
```

---

## 📁 Project Structure

```
Music-Popularity-Prediction/
│
├── main.py
├── songs.csv
├── README.md
└── requirements.txt
```

---

## 🧾 Results Summary

* Baseline Accuracy (Logistic Regression): ~63%
* Improved Accuracy (Random Forest): ~75%
* Model is balanced and realistic
* Results are suitable for real-world decision support

---

## 🎓 What I Learned from This Project

* End-to-end data science workflow
* Data cleaning and preprocessing
* Feature engineering techniques
* Model training and evaluation
* Model comparison and improvement
* Interpreting results and explaining insights

---

## 🧑‍💼 Resume Description

**Music Popularity Prediction using Machine Learning**

* Built an end-to-end ML pipeline to predict song popularity using audio features
* Performed data cleaning, feature engineering, and exploratory data analysis
* Trained Logistic Regression and Random Forest models
* Improved prediction accuracy and analyzed feature importance

---

## 📌 Future Improvements

* Apply NLP techniques on lyrics
* Perform hyperparameter tuning
* Deploy model as a web application
* Predict popularity score instead of binary output

