import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import time
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="GymWolf - Fitness Tracker", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# UI styling
st.markdown("""
<style>
    /* Main colors */
    :root {
        --gymwolf-blue: #262626;
        --gymwolf-dark: #262626;
        --gymwolf-light: #f6f6f8;
        --gymwolf-gray: #aaa29d;
    }
    
    /* Header/Navigation styling */
    .stApp header {
        background-color: var(--gymwolf-dark) !important;
        color: white !important;
    }
    
    /* Main app styling */
    .main {
        background-color: black;
    }
    
    /* Custom header */
    .gymwolf-header {
        background-color: var(--gymwolf-dark);
        color: white;
        padding: 15px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .gymwolf-logo {
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        font-size: 24px;
        color: white;
    }
    
    /* Hero section */
.hero-section {
    animation: slideBg 15s infinite;
    background-size: cover;
    background-position: center;
    color: black;
    padding: 80px 40px;
    text-align: center;
    border-radius: 10px;
    margin-bottom: 30px;
}

/* Slideshow animation */
@keyframes slideBg {
    0% {
        background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0)), 
        url('https://downloads.ctfassets.net/6ilvqec50fal/3RDPHNmSlJLFcUkgfzLNVd/d84e476deab8b23e936baf7cc4631bde/Bear_Walkout.gif');
    }
    100% {
        background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0)), 
        url('https://www.tonal.com/wp-content/uploads/2022/06/18-Workouts-Hero.gif?fit=1200%2C800');
    }
}

    
    .hero-title {
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    .hero-subtitle {
        font-size: 20px;
        margin-bottom: 30px;
    }
    
    /* Button styling */
    .gymwolf-button {
        background-color: var(--gymwolf-blue);
        color: black;
        border: none;
        padding: 10px 30px;
        font-size: 18px;
        border-radius: 10px;
        cursor: pointer;
        text-align: center;
        text-decoration: none;
        display: inline-block;
    }
    
    /* Feature section */
    .feature-section {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
    }
    
    .feature-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
        color: var(--gymwolf-dark);
    }
    
    .feature-description {
        font-size: 16px;
        margin-bottom: 15px;
    }
    
    /* Form styling */
    .stTextInput, .stNumberInput, .stSelectbox, .stSlider {
        margin-bottom: 20px;
    }
    
    .stButton>button {
        background-color: var(--gymwolf-blue);
        color: black;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 24px;
    }
    
    /* Footer */
    .footer {
        background-color: var(--gymwolf-blue);
        color: white;
        padding: 20px 0;
        text-align: center;
        margin-top: 50px;
    }
    
    .copyright {
        font-size: 14px;
    }
    
    /* Card styling */
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Clean radio button styling */
    div[role="radiogroup"] label {
        background-color: transparent !important;
        padding: 0px;
        margin-right: 8px;
        border-radius: 0;
        box-shadow: none;
}

    
    /* Progress info styling */
    .progress-info {
        padding: 15px;
        border-left: 4px solid var(--gymwolf-blue);
        background-color: #f0f7fb;
        margin-bottom: 15px;
    }
    
    /* Insights box */
    .insights-box {
        padding: 20px;
        margin-top: 20px;
        margin-bottom: 20px;
        background: #ede9fe;
        border-left: 5px solid #7c3aed;
    }
</style>
""", unsafe_allow_html=True)

# Custom header
st.markdown("""
<div class="gymwolf-header">
    <div class="gymwolf-logo">ROYAL FITNESS</div>
    <div>
        <span style="margin-right: 15px;">Register</span>
        <span>Sign In</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">A better way to track your workouts</div>
    <div class="hero-subtitle">Track your progress, achieve your fitness goals</div>
    <button class="gymwolf-button">Start Free</button>
</div>
""", unsafe_allow_html=True)

data = pd.read_csv(r"C:\Users\sunil\Downloads\Major project (1).csv", parse_dates=['date'])
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
data.drop_duplicates(inplace=True)
data.sort_index(inplace=True)
data = data.drop(columns='participant_id')

# Fill missing health_condition values
def fill_health_condition(row):
    if pd.isna(row['health_condition']):
        group = data[
            (data['smoking_status'] == row['smoking_status']) &
            (data['stress_level'] == row['stress_level']) &
            (data['bmi'].between(row['bmi'] - 1, row['bmi'] + 1)) &
            (data['age'].between(row['age'] - 3, row['age'] + 3))
        ]
        if not group['health_condition'].dropna().empty:
            return group['health_condition'].dropna().mode()[0]
        else:
            return data['health_condition'].dropna().mode()[0]
    else:
        return row['health_condition']

data['health_condition'] = data.apply(fill_health_condition, axis=1)
data.dropna(subset=['calories_burned'], inplace=True)


# Encoding categorical variables
data['gender'] = data['gender'].map({'M': 1, 'F': 0})
data['smoking_status'] = data['smoking_status'].map({'Never': 0, 'Former': 1, 'Current': 2})
data['health_condition'] = data['health_condition'].map({'Healthy': 0, 'Diabetes': 1, 'Hypertension': 2, 'Heart Disease': 3})
data['activity_type'] = data['activity_type'].map({
    'Running': 0, 'Cycling': 1, 'Swimming': 2, 'Walking': 3, 'Yoga': 4,
    'Weight Training': 5, 'Basketball': 6, 'Dancing': 7, 'HIIT': 8})


# Features for training
features = [
    "age","gender", "height_cm", "weight_kg", "bmi", "smoking_status",
    "activity_type", "hours_sleep", "stress_level", "health_condition"
]

X = data[features]
y = data['calories_burned']

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
model.fit(X_train, y_train)



def user_data():
    age = st.sidebar.number_input("Age", 10, 100)
    gender = st.sidebar.radio("Gender", ["Male", "Female"])
    height_cm = st.sidebar.number_input("Height (cm)", 130, 210, 10)
    weight_kg = st.sidebar.number_input("Weight (kg)", 30, 150, 10)
    bmi = weight_kg / ((height_cm / 100) ** 2)
    smoking = st.sidebar.selectbox("Smoking Status", ["Never", "Former", "Current"])
    activity = st.sidebar.selectbox("Activity Type", ["Running", "Cycling", "Swimming", "Walking", "Yoga", "Weight Training", "Basketball", "Dancing", "HIIT"])
    sleep = st.sidebar.number_input("Sleep Duration (hrs)", 4.0, 12.0, 0.5)
    stress = st.sidebar.number_input("Stress Level", 0, 10, 1)
    health = st.sidebar.selectbox("Health Condition", ["Healthy", "Diabetes", "Hypertension", "Heart Disease"])


    return pd.DataFrame({
        "age": [age],
        "gender": [1 if gender == "Male" else 0],
        "height_cm": [height_cm],
        "weight_kg": [weight_kg],
        "bmi": [bmi],
        "smoking_status": [{"Never": 0, "Former": 1, "Current": 2}[smoking]],
        "activity_type": [{"Running": 0, "Cycling": 1, "Swimming": 2, "Walking": 3, "Yoga": 4, "Weight Training": 5, "Basketball": 6, "Dancing": 7, "HIIT": 8}[activity]],
        "hours_sleep": [sleep],
        "stress_level": [stress],
        "health_condition": [{"Healthy": 0, "Diabetes": 1, "Hypertension": 2, "Heart Disease": 3}[health]]
    })



st.markdown('<div class="feature-section">', unsafe_allow_html=True)
st.write('## Track Your Gym and Fitness Progress')
st.write('Royal Fitness empowers your fitness journey. Track every rep, every step, and every workout. Get smart calorie burn predictions with powerful insights. Visualize your progress and crush your goals. Your personal trainer, right in your pocket.')

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('### Enter Your Information')
    
    with st.form("user_info_form"):
        age = st.number_input("Age", 10, 100, 30)
        gender = st.radio("Gender", ["Male", "Female"])
        height_cm = st.number_input("Height (cm)", 130, 210, 170)
        weight_kg = st.number_input("Weight (kg)", 30, 150, 70)
        bmi = weight_kg / ((height_cm / 100) ** 2)
        st.write(f"Your BMI: **{bmi:.2f}**")
        
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        activity = st.selectbox("Activity Type", ["Running", "Cycling", "Swimming", "Walking", "Yoga", 
                                                 "Weight Training", "Basketball", "Dancing", "HIIT"])
        sleep = st.slider("Sleep Duration (hrs)", 4.0, 12.0, 7.0, 0.1)
        stress = st.slider("Stress Level", 0, 10, 3)
        health = st.selectbox("Health Condition", ["Healthy", "Diabetes", "Hypertension", "Heart Disease"])
        
        submit_button = st.form_submit_button("Calculate Calories Burned")
    
    # Convert inputs to model format
    user_data = pd.DataFrame({
        "Age": [age],
        "Gender": [1 if "Male" else 0],
        "Height(cm)": [height_cm],
        "Weight(kg)": [weight_kg],
        "BMI": [bmi],
        "Smoking Status": [{"Never" : 0, "Former" :1, "Current": 2}[smoking]],
        "Activity Type": [{"Running": 0, "Cycling" :1, "Swimming": 2, "Walking": 3, "Yoga": 4, 
                           "Weight Training": 5, "Basketball": 6, "Dancing": 7, "HIIT": 8}[activity]],
        "Sleep Duration": [sleep],
        "Stress Level": [stress],
        "Health Condition": [{"Healthy":0, "Diabetes":1, "Hypertension":2, "Heart Disease":3}[health]]
    })

with col2:
    st.markdown('### Your Results')
    
    if submit_button:  # Changed from 'submit_button' in locals()
        st.markdown('<div class="progress-info">', unsafe_allow_html=True)
        with st.spinner("Analyzing your input and predicting calories burned..."):
            time.sleep(1)  # Simulate processing
            prediction = model.predict(user_data)[0]
        
        st.success(f"You burned approximately **{round(prediction, 2)} kilocalories** during your activity.")
        
        # Create a gauge chart for calories burned
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Calories Burned"},
            gauge={
                'axis': {'range': [None, 1500]},
                'bar': {'color': "#22a6e0"},
                'steps': [
                    {'range': [0, 500], 'color': "#e8f7fd"},
                    {'range': [500, 1000], 'color': "#a8e0f7"},
                    {'range': [1000, 1500], 'color': "#65c9f1"}
                ]
            }
        ))
        st.plotly_chart(fig)
        
        # Similar users section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### People with Similar Stats")
        similar_range = data[(data['calories_burned'] >= prediction - 100) & 
                             (data['calories_burned'] <= prediction + 100)]
        
        if not similar_range.empty:
            st.dataframe(similar_range.sample(min(5, len(similar_range))))
        else:
            st.info("No similar users found in this range.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature importance (insights)
        st.markdown('<div class="insights-box">', unsafe_allow_html=True)
        st.markdown("#### Fitness Insights")
        
        # Calculate personalized insights
        older = (data['age'] > user_data['age'].values[0]).mean() * 100
        active = (data['activity_type'] > user_data['activity_type'].values[0]).mean() * 100
        
        st.write(f"- You are older than **{older:.1f}%** of participants.")
        st.write(f"- You exercised more actively than **{active:.1f}%** of people.")
        
        # Add a fitness tip
        st.markdown("**Fitness Tip:**")
        st.write("Consistency matters more than intensity. Regular workouts combined with hydration, balanced diet, and sleep can create sustainable fitness over time.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Fill out the form and click 'Calculate Calories Burned' to see your results.")
        
        # Sample visualization
        st.markdown("#### Sample Fitness Progress")
        
        # Create sample data for plot
        days = pd.date_range(start='2023-01-01', periods=30)
        calories = [np.random.randint(300, 800) for _ in range(30)]
        
        fig = px.line(
            x=days, y=calories, 
            labels={"x": "Date", "y": "Calories Burned"},
            title="Example Progress Chart"
        )
        fig.update_traces(line_color="#22a6e0")
        st.plotly_chart(fig)

# (Rest of the code remains the same)

st.markdown('</div>', unsafe_allow_html=True)



# Database section
st.markdown("""
< style="padding: 20px 0;">
        <h2 style="font-size: 24px; font-weight: bold; text-align: center;">Database with 300+ Exercises</h2>
        <p style="font-size: 16px; text-align: center; max-width: 800px; margin: 20px auto;">
        Our app has a database with over 300 exercises. The exercises come with detailed descriptions, 
        tips, and step-by-step images. If you don't find an exercise from our database, you can also add your own exercises.
        </p>
""", unsafe_allow_html=True)


# Footer section
st.markdown("""
<div style="background-color: #282828; color: white; padding: 20px 0;">
    <div style="display: flex; justify-content: space-around; flex-wrap: wrap; max-width: 800px; margin: 0 auto; text-align: left;">
        <div style="margin: 10px 20px;">
            <p style="font-weight: bold; font-size: 16px;">ABOUT</p>
            <p>About us</p>
            <p>Contact</p>
        </div>
        <div style="margin: 10px 20px;">
            <p style="font-weight: bold; font-size: 16px;">LEGAL</p>
            <p>Privacy Policy</p>
            <p>Terms of Service</p>
            <p>Disclaimer</p>
        </div>
        <div style="margin: 10px 20px;">
            <p style="font-weight: bold; font-size: 16px;">SUPPORT</p>
            <p>Help Center</p>
            <p>FAQs</p>
            <p>Contact us</p>
            <p>Feedback</p>
        </div>
    </div>
            <div style="margin: 10px 20px; text-align: center;">
            <p>Created By Sunil Naik with ❤️</p>
            </div>

</div>

""", unsafe_allow_html=True)
