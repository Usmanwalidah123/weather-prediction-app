import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/weather-nominal-weka.csv"
    return pd.read_csv(url)

df = load_data()
st.title("Weather Prediction App")
st.write("### Raw Dataset")
st.dataframe(df)

# Encoding
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)
X = df_encoded.drop('play', axis=1)
y = df_encoded['play']

# Mapping back original values for dropdowns
feature_options = {col: df[col].unique().tolist() for col in df.columns if col != 'play'}

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Train models
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
rf = RandomForestClassifier(max_depth=2, random_state=100)
rf.fit(X_train, y_train)

# User input section
st.write("## Make a Prediction")

# Collect input for each feature
user_input = {}
for feature, options in feature_options.items():
    user_input[feature] = st.selectbox(f"{feature.capitalize()}", options)

# Convert user input to encoded format
user_df = pd.DataFrame([user_input])
user_encoded = user_df.apply(lambda col: le.fit(df[col.name]).transform(col))

# Select model
model_choice = st.radio("Choose Model", ("Decision Tree", "Random Forest"))

# Predict button
if st.button("Predict"):
    if model_choice == "Decision Tree":
        prediction = dt.predict(user_encoded)[0]
    else:
        prediction = rf.predict(user_encoded)[0]

    # Decode target label
    predicted_label = df['play'].unique()[prediction]
    
    st.success(f"Prediction: {predicted_label}")
