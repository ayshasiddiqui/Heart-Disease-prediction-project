import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<style>

.main{
background-color:#f7f9fc;
}

h1,h2,h3{
color:#d62828;
}

.stButton>button{
background:#d62828;
color:white;
border-radius:10px;
height:45px;
width:220px;
font-size:18px;
}

.stButton>button:hover{
background:#003566;
color:white;
}

.css-1d391kg{
background:#edf2fb;
}

</style>
""",unsafe_allow_html=True)

@st.cache_data
def load_data():
    df=pd.read_csv("data.csv")
    return df

df=load_data()

try:
    model=joblib.load("model.pkl")
except:
    model=None

st.sidebar.title("❤️ Heart Disease Prediction")

page=st.sidebar.radio(
"Navigation",
[
"Home",
"Dashboard",
"Visualizations",
"Prediction",
"About"
]
)

if page=="Home":

    st.title("❤️ Heart Disease Prediction System")

    st.write("---")

    st.image(
    "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=1200",
    use_container_width=True
    )

    st.markdown("""
### About Project

This Machine Learning project predicts whether a patient has Heart Disease or not.

### Algorithms Used

- Logistic Regression
- Decision Tree
- Random Forest

### Technologies

- Python
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- Scikit Learn

""")

    c1,c2,c3,c4=st.columns(4)

    c1.metric(
        "Total Patients",
        len(df)
    )

    c2.metric(
        "Average Age",
        round(df["Age"].mean(),2)
    )

    c3.metric(
        "Average Cholesterol",
        round(df["Cholesterol"].mean(),2)
    )

    c4.metric(
        "Average Heart Rate",
        round(df["Heart Rate"].mean(),2)
    )

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10),use_container_width=True)

elif page=="Dashboard":

    st.title("📊 Dashboard")

    st.write("---")

    col1,col2=st.columns(2)

    with col1:

        fig,ax=plt.subplots(figsize=(6,4))

        sns.countplot(
            x="Heart Disease",
            data=df,
            palette="Set2",
            ax=ax
        )

        ax.set_title("Heart Disease Distribution")

        st.pyplot(fig)

    with col2:

        fig,ax=plt.subplots(figsize=(6,4))

        sns.histplot(
            df["Age"],
            kde=True,
            color="skyblue",
            ax=ax
        )

        ax.set_title("Age Distribution")

        st.pyplot(fig)

    col3,col4=st.columns(2)

    with col3:

        fig,ax=plt.subplots(figsize=(6,4))

        sns.countplot(
            x="Gender",
            data=df,
            palette="pastel",
            ax=ax
        )

        ax.set_title("Gender Distribution")

        st.pyplot(fig)

    with col4:

        fig,ax=plt.subplots(figsize=(6,4))

        sns.histplot(
            df["Cholesterol"],
            kde=True,
            color="orange",
            ax=ax
        )

        ax.set_title("Cholesterol Distribution")

        st.pyplot(fig)

    col5,col6=st.columns(2)

    with col5:

        fig,ax=plt.subplots(figsize=(6,4))

        sns.histplot(
            df["Blood Pressure"],
            kde=True,
            color="green",
            ax=ax
        )

        ax.set_title("Blood Pressure")

        st.pyplot(fig)

    with col6:

        fig,ax=plt.subplots(figsize=(6,4))

        sns.histplot(
            df["Heart Rate"],
            kde=True,
            color="red",
            ax=ax
        )

        ax.set_title("Heart Rate")

        st.pyplot(fig)

elif page=="Visualizations":

    st.title("📈 Data Visualizations")

    st.write("---")
    col1,col2=st.columns(2)

    with col1:

        fig,ax=plt.subplots(figsize=(6,4))

        sns.countplot(
            x="Smoking",
            data=df,
            palette="viridis",
            ax=ax
        )

        ax.set_title("Smoking Status")

        st.pyplot(fig)

    with col2:

        fig,ax=plt.subplots(figsize=(6,4))

        sns.countplot(
            x="Diabetes",
            data=df,
            palette="Set2",
            ax=ax
        )

        ax.set_title("Diabetes")

        st.pyplot(fig)


    col3,col4=st.columns(2)

    with col3:

        fig,ax=plt.subplots(figsize=(6,4))

        sns.countplot(
            x="Alcohol Intake",
            data=df,
            palette="rocket",
            ax=ax
        )

        ax.set_title("Alcohol Intake")

        st.pyplot(fig)

    with col4:

        fig,ax=plt.subplots(figsize=(6,4))

        sns.countplot(
            x="Chest Pain Type",
            data=df,
            palette="coolwarm",
            ax=ax
        )

        plt.xticks(rotation=20)

        ax.set_title("Chest Pain Type")

        st.pyplot(fig)


    col5,col6=st.columns(2)

    with col5:

        fig,ax=plt.subplots(figsize=(6,4))

        sns.histplot(
            df["Exercise Hours"],
            kde=True,
            color="purple",
            ax=ax
        )

        ax.set_title("Exercise Hours")

        st.pyplot(fig)

    with col6:

        fig,ax=plt.subplots(figsize=(6,4))

        sns.histplot(
            df["Blood Sugar"],
            kde=True,
            color="red",
            ax=ax
        )

        ax.set_title("Blood Sugar")

        st.pyplot(fig)


    st.subheader("Heart Disease Percentage")

    fig,ax=plt.subplots(figsize=(6,6))

    df["Heart Disease"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax
    )

    ax.set_ylabel("")

    st.pyplot(fig)
    

    st.subheader("Box Plot")

    fig,ax=plt.subplots(figsize=(10,5))

    sns.boxplot(
        data=df[
            [
                "Age",
                "Cholesterol",
                "Blood Pressure",
                "Heart Rate"
            ]
        ],
        ax=ax
    )

    st.pyplot(fig)

elif page=="Prediction":

    st.title("❤️ Heart Disease Prediction")

    st.write("---")

    age=st.number_input("Age",18,100,30)

    gender=st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    cholesterol=st.number_input(
        "Cholesterol",
        100,
        400,
        200
    )

    bp=st.number_input(
        "Blood Pressure",
        80,
        220,
        120
    )

    heart_rate=st.number_input(
        "Heart Rate",
        40,
        200,
        80
    )

    smoking=st.selectbox(
        "Smoking",
        ["Yes","No"]
    )

    alcohol=st.selectbox(
        "Alcohol Intake",
        ["Yes","No"]
    )

    exercise=st.slider(
        "Exercise Hours",
        0,
        20,
        5
    )

    family=st.selectbox(
        "Family History",
        ["Yes","No"]
    )

    diabetes=st.selectbox(
        "Diabetes",
        ["Yes","No"]
    )

    obesity=st.selectbox(
        "Obesity",
        ["Yes","No"]
    )

    stress=st.slider(
        "Stress Level",
        1,
        10,
        5
    )

    sugar=st.number_input(
        "Blood Sugar",
        70,
        300,
        120
    )

    angina=st.selectbox(
        "Exercise Induced Angina",
        ["Yes","No"]
    )

    chest=st.selectbox(
        "Chest Pain Type",
        [
            "Typical",
            "Atypical",
            "Non-anginal",
            "Asymptomatic"
        ]
    )
    if st.button("Predict Heart Disease"):

        gender = 1 if gender == "Male" else 0
        smoking = 1 if smoking == "Yes" else 0
        alcohol = 1 if alcohol == "Yes" else 0
        family = 1 if family == "Yes" else 0
        diabetes = 1 if diabetes == "Yes" else 0
        obesity = 1 if obesity == "Yes" else 0
        angina = 1 if angina == "Yes" else 0

        chest_dict = {
            "Typical":0,
            "Atypical":1,
            "Non-anginal":2,
            "Asymptomatic":3
        }

        chest = chest_dict[chest]

        input_data = np.array([[
            age,
            gender,
            cholesterol,
            bp,
            heart_rate,
            smoking,
            alcohol,
            exercise,
            family,
            diabetes,
            obesity,
            stress,
            sugar,
            angina,
            chest
        ]])

        prediction = model.predict(input_data)

        st.write("---")

        if prediction[0] == 1:
            st.error("⚠ High Risk of Heart Disease")
        else:
            st.success("✅ No Heart Disease Detected")

elif page == "About":

    st.title("ℹ About Project")

    st.write("---")

    st.markdown("""

# ❤️ Heart Disease Prediction

This project predicts whether a patient has Heart Disease using Machine Learning.

## Algorithms Used

- Logistic Regression
- Decision Tree
- Random Forest

## Libraries Used

- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset Features

- Age
- Gender
- Cholesterol
- Blood Pressure
- Heart Rate
- Smoking
- Alcohol Intake
- Exercise Hours
- Family History
- Diabetes
- Obesity
- Stress Level
- Blood Sugar
- Exercise Induced Angina
- Chest Pain Type

Target Variable

Heart Disease

""")

    st.success("Project Developed by Sanobar Hayat ❤️")

st.write("---")
st.caption("© 2026 Heart Disease Prediction System | Streamlit Project")