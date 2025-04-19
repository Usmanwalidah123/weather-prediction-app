import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/weather-nominal-weka.csv"
    return pd.read_csv(url)

df = load_data()
st.title("Weather Prediction App 🌦️")
st.write("## Raw Dataset")
st.dataframe(df)

# Encoding
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)
X = df_encoded.drop('play', axis=1)
y = df_encoded['play']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Train models
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
rf = RandomForestClassifier(max_depth=2, random_state=100)
rf.fit(X_train, y_train)

# Predictions
y_dt_pred_train = dt.predict(X_train)
y_dt_pred_test = dt.predict(X_test)
y_rf_pred_train = rf.predict(X_train)
y_rf_pred_test = rf.predict(X_test)

# Accuracy
st.write("## Model Accuracy")
results = {
    'Decision Tree': {
        'Train Accuracy': accuracy_score(y_train, y_dt_pred_train),
        'Test Accuracy': accuracy_score(y_test, y_dt_pred_test)
    },
    'Random Forest': {
        'Train Accuracy': accuracy_score(y_train, y_rf_pred_train),
        'Test Accuracy': accuracy_score(y_test, y_rf_pred_test)
    }
}
results_df = pd.DataFrame(results)
st.dataframe(results_df)

# Tree Visualization
st.write("## Decision Tree")
fig, ax = plt.subplots(figsize=(10, 6))
plot_tree(dt, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], ax=ax)
st.pyplot(fig)

# Classification Report
st.write("## Classification Report")
st.text("Decision Tree\n" + classification_report(y_test, y_dt_pred_test))
st.text("Random Forest\n" + classification_report(y_test, y_rf_pred_test))

# Scatter Plot
st.write("## Actual vs Predicted (Decision Tree)")
fig2, ax2 = plt.subplots()
ax2.scatter(y_train, y_dt_pred_train, alpha=0.5)
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")
ax2.set_title("Decision Tree Prediction")
st.pyplot(fig2)
