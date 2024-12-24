import streamlit as st
import pickle
import numpy as np
import pandas as pd
import google.generativeai as genai
import re
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load necessary files
scaler = pickle.load(open("scaler.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))
linkedin_df = pd.read_csv("LinkedIn people profiles datasets.csv")

# Google Gemini API Configuration
API_KEY = "AIzaSyA_wp4-mnq_ntJNtfh77XmRkCrxkq45zdQ"  # Replace with your actual API key
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Class names for career prediction
class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']

def recommend_by_marks(gender, part_time_job, absence_days, extracurricular_activities,
                       weekly_self_study_hours, math_score, history_score, physics_score,
                       chemistry_score, biology_score, english_score, geography_score):
    # Calculate total and average scores
    total_score = sum([math_score, history_score, physics_score, chemistry_score,
                       biology_score, english_score, geography_score])
    average_score = total_score / 7

    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_encoded = 1 if extracurricular_activities else 0

    # Create feature array
    feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days,
                               extracurricular_encoded, weekly_self_study_hours,
                               math_score, history_score, physics_score, chemistry_score,
                               biology_score, english_score, geography_score,
                               total_score, average_score]])

    # Scale features and predict
    scaled_features = scaler.transform(feature_array)
    probabilities = model.predict_proba(scaled_features)
    top_classes_idx = np.argsort(-probabilities[0])[:5]
    
    return [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]

def recommend_linkedin_profiles(prediction):
    matching_profiles = linkedin_df[linkedin_df['position'].str.contains(prediction, case=False, na=False)]
    if not matching_profiles.empty:
        return matching_profiles.sample(n=min(5, len(matching_profiles)))
    return pd.DataFrame()

def upload_and_parse_resume(uploaded_file):
    try:
        # Save the uploaded file temporarily
        with open("temp_resume.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Use Gemini to parse the resume
        sample_file = genai.upload_file(path="temp_resume.pdf", display_name="Resume")
        
        response = gemini_model.generate_content([
            sample_file,
            "Extract the following details from the resume: "
            "gender, part-time job status, number of absence days, "
            "extracurricular activities, weekly self-study hours, "
            "math score, history score, physics score, chemistry score, "
            "biology score, English score, geography score."
        ])
        
        parsed_data = response.text
        return extract_fields_from_response(parsed_data)

    except Exception as e:
        st.error(f"Error during resume parsing: {e}")
        return None

def extract_fields_from_response(response_text):
    try:
        parsed_resume = {
            "gender": re.search(r"\*\*Gender:\*\* ([^\n]*)", response_text).group(1).strip(),
            "part_time_job": "yes" in re.search(r"\*\*Part-Time Job:\*\* ([^\n]*)", response_text).group(1).strip().lower(),
            "absence_days": int(re.search(r"\*\*Absence Days:\*\* (\d+)", response_text).group(1)),
            "extracurricular": "yes" in re.search(r"\*\*Extracurricular Activities:\*\* ([^\n]*)", response_text).group(1).strip().lower(),
            "self_study_hours": int(re.search(r"\*\*Weekly Self-Study Hours:\*\* (\d+)", response_text).group(1)),
            "math_score": int(re.search(r"\*\*Math Score:\*\* (\d+)", response_text).group(1)),
            "history_score": int(re.search(r"\*\*History Score:\*\* (\d+)", response_text).group(1)),
            "physics_score": int(re.search(r"\*\*Physics Score:\*\* (\d+)", response_text).group(1)),
            "chemistry_score": int(re.search(r"\*\*Chemistry Score:\*\* (\d+)", response_text).group(1)),
            "biology_score": int(re.search(r"\*\*Biology Score:\*\* (\d+)", response_text).group(1)),
            "english_score": int(re.search(r"\*\*English Score:\*\* (\d+)", response_text).group(1)),
            "geography_score": int(re.search(r"\*\*Geography Score:\*\* (\d+)", response_text).group(1))
        }
        return parsed_resume

    except Exception as e:
        st.error(f"Error extracting fields from response: {e}")
        return None

def main():
    st.set_page_config(page_title="Career Recommendation System", page_icon=":rocket:")
    
    st.title("Career Recommendation System")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose Recommendation Method", 
                                ["Home", "Recommend by Marks", "Recommend by Resume"])
    
    if page == "Home":
        st.write("## Welcome to Career Recommendation System")
        st.write("Choose a recommendation method from the sidebar:")
        st.write("- **Recommend by Marks**: Input your academic scores")
        st.write("- **Recommend by Resume**: Upload your resume for analysis")
    
    elif page == "Recommend by Marks":
        st.header("Career Recommendation through Marks")
        
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            part_time_job = st.checkbox("Part-time Job")
            absence_days = st.number_input("Absence Days", min_value=0, max_value=30, value=0)
            extracurricular_activities = st.checkbox("Extracurricular Activities")
            weekly_self_study_hours = st.number_input("Weekly Self-Study Hours", min_value=0, max_value=20, value=0)
        
        with col2:
            math_score = st.number_input("Math Score", min_value=0, max_value=100, value=0)
            history_score = st.number_input("History Score", min_value=0, max_value=100, value=0)
            physics_score = st.number_input("Physics Score", min_value=0, max_value=100, value=0)
            chemistry_score = st.number_input("Chemistry Score", min_value=0, max_value=100, value=0)
            biology_score = st.number_input("Biology Score", min_value=0, max_value=100, value=0)
            english_score = st.number_input("English Score", min_value=0, max_value=100, value=0)
            geography_score = st.number_input("Geography Score", min_value=0, max_value=100, value=0)
        
        if st.button("Get Recommendations"):
            recommendations = recommend_by_marks(
                gender, part_time_job, absence_days, extracurricular_activities,
                weekly_self_study_hours, math_score, history_score, physics_score,
                chemistry_score, biology_score, english_score, geography_score
            )
            
            st.subheader("Top Career Recommendations")
            for career, probability in recommendations:
                st.write(f"{career}: {probability:.2%}")
            
            # Get LinkedIn profiles for top recommendation
            top_career = recommendations[0][0]
            linkedin_profiles = recommend_linkedin_profiles(top_career)
            
            if not linkedin_profiles.empty:
                st.subheader(f"LinkedIn Profiles for {top_career}")
                st.dataframe(linkedin_profiles)
    
    elif page == "Recommend by Resume":
        st.header("Career Recommendation through Resume Parsing")
        
        uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        
        if uploaded_file is not None:
            st.write("Resume uploaded successfully!")
            
            if st.button("Parse Resume and Get Recommendations"):
                # Parse resume
                parsed_resume = upload_and_parse_resume(uploaded_file)
                
                if parsed_resume:
                    st.subheader("Extracted Resume Details")
                    for key, value in parsed_resume.items():
                        st.write(f"{key.replace('_', ' ').title()}: {value}")
                    
                    # Get career recommendations
                    recommendations = recommend_by_marks(
                        gender=parsed_resume['gender'],
                        part_time_job=parsed_resume['part_time_job'],
                        absence_days=parsed_resume['absence_days'],
                        extracurricular_activities=parsed_resume['extracurricular'],
                        weekly_self_study_hours=parsed_resume['self_study_hours'],
                        math_score=parsed_resume['math_score'],
                        history_score=parsed_resume['history_score'],
                        physics_score=parsed_resume['physics_score'],
                        chemistry_score=parsed_resume['chemistry_score'],
                        biology_score=parsed_resume['biology_score'],
                        english_score=parsed_resume['english_score'],
                        geography_score=parsed_resume['geography_score']
                    )
                    
                    st.subheader("Top Career Recommendations")
                    for career, probability in recommendations:
                        st.write(f"{career}: {probability:.2%}")
                    
                    # Get LinkedIn profiles for top recommendation
                    top_career = recommendations[0][0]
                    linkedin_profiles = recommend_linkedin_profiles(top_career)
                    
                    if not linkedin_profiles.empty:
                        st.subheader(f"LinkedIn Profiles for {top_career}")
                        st.dataframe(linkedin_profiles)

if __name__ == "__main__":
    main()