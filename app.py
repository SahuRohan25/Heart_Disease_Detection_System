import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Heart Disease Detection", layout="wide")

st.title("❤️ Heart Disease Detection App")
st.write("This app predicts whether a patient has heart disease based on medical data.")

# Load model and scaler
@st.cache_resource
def load_artifacts():
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# Sidebar options
st.sidebar.header("Input Patient Data")
upload_option = st.sidebar.radio("Choose data input method:", ("Upload CSV", "Manual Entry"))

if upload_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your heart.csv file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview", df.head())

        if "target" not in df.columns:
            X_scaled = scaler.transform(df)  # use pre-trained scaler

            if df.shape[0] == 1:
                pred = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled)[0][1]
                if pred == 1:
                    st.error(f"⚠️ Heart Disease Detected (Probability: {prob:.2f})")
                else:
                    st.success(f"✅ No Heart Disease Detected (Probability: {prob:.2f})")
            else:
                preds = model.predict(X_scaled)
                probs = model.predict_proba(X_scaled)[:, 1]
                results = pd.DataFrame({"Prediction": preds, "Probability": probs})
                st.write("### Predictions", results)
        else:
            st.success("Target column found. This file contains labels, so predictions are skipped.")

elif upload_option == "Manual Entry":
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.sidebar.selectbox("Sex", [0, 1])
    cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.sidebar.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.sidebar.selectbox("Slope of peak exercise ST segment", [0, 1, 2])
    ca = st.sidebar.selectbox("Number of major vessels (0-3) colored by flourosopy", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)", [0, 1, 2])

    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                 thalach, exang, oldpeak, slope, ca, thal]],
                               columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    if st.sidebar.button("Predict"):
        X_scaled = scaler.transform(input_data)
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        if prediction == 1:
            st.error(f"⚠️ Heart Disease Detected (Probability: {probability:.2f})")
        else:
            st.success(f"✅ No Heart Disease Detected (Probability: {probability:.2f})")

# EDA Section
st.markdown("---")
st.subheader("📊 Exploratory Data Analysis (EDA)")

eda_file = st.file_uploader("Upload CSV for EDA", type=["csv"])
if eda_file is not None:
    eda_df = pd.read_csv(eda_file)

    if "target" in eda_df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x='target', data=eda_df, ax=ax)
        ax.set_title("Heart Disease Distribution")
        st.pyplot(fig)

    if eda_df.shape[0] > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(eda_df.corr(), annot=True, ax=ax)
        ax.set_title("Feature Correlations")
        st.pyplot(fig)
    else:
        st.info("Correlation heatmap not shown because dataset has only one record.")
        st.write("### Patient Data", eda_df)

        if "target" not in eda_df.columns:
            X_scaled = scaler.transform(eda_df)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1]
            if prediction == 1:
                st.error(f"⚠️ Heart Disease Detected (Probability: {probability:.2f})")
            else:
                st.success(f"✅ No Heart Disease Detected (Probability: {probability:.2f})")
        else:
            st.warning("Target column found in the file — skipping prediction.")
