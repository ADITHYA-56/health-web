import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Corrected dataset
data = {
    'Category': [
        'Fitness', 'Health', 'Nutrition', 'Medicine', 'Fitness', 'Health', 
        'Nutrition', 'Medicine', 'Fitness', 'Health', 'Nutrition', 'Medicine',
        'Fitness', 'Health', 'Nutrition', 'Medicine', 'Fitness', 'Health', 
        'Nutrition', 'Medicine', 'Fitness', 'Health', 'Nutrition', 'Medicine'
    ],
    
    'Title': [
        'Yoga for Beginners', 'Managing Stress', 'Balanced Diet', 'Headache', 
        'Strength Training', 'Mental Wellness', 'Protein Intake', 'Cold Medication',
        'Cardio for Endurance', 'Sleep Hygiene', 'Vitamin D Importance', 
        'Heart Disease Prevention', 'HIIT for Weight Loss', 'Blood Pressure Control', 
        'Low Carb Diet', 'Antibiotics Use', 'Pilates for Core Strength', 'Diabetes Management',
        'Omega-3 Benefits', 'Asthma Treatment', 'Flexibility Training', 'Healthy Aging Tips',
        'Fiber-Rich Foods', 'Skin Allergy Relief'
    ],

    'Description': [
        'Yoga and breathing exercises for beginners',
        'Techniques for reducing stress and improving mental health',
        'eat at least 5 portions of a variety of fruit and vegetables every day',
        'Remedies that may reduce headache pain include aspirin, paracetamol and ibuprofen. Resting in a darkened room may also help.',
        'A complete guide to strength training and muscle building',
        'Ways to improve mental wellness and reduce anxiety',
        'Include protein in your diet for better health',
        'Common medication for treating colds and flu symptoms',
        'Cardiovascular exercises to improve endurance and stamina',
        'Tips for maintaining good sleep hygiene and restful sleep',
        'The role of Vitamin D in maintaining bone and immune health',
        'Steps to prevent heart diseases through lifestyle changes',
        'High-Intensity Interval Training for effective weight loss',
        'Methods to control blood pressure naturally',
        'The benefits of low carbohydrate diets for weight management',
        'Understanding when and how to use antibiotics effectively',
        'Pilates exercises to strengthen core muscles and improve posture',
        'Guidelines for managing diabetes through diet and medication',
        'Health benefits of Omega-3 fatty acids for brain and heart',
        'Treatments for managing asthma symptoms and prevention',
        'Exercises to improve flexibility and prevent injuries',
        'Tips for healthy aging, including diet, exercise, and lifestyle',
        'The importance of fiber-rich foods for digestion and heart health',
        'Medications and treatments for common skin allergies'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define TF-IDF Vectorizer and fit it on all data descriptions
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Description'])

# Streamlit UI for user inputs
st.title('Health Monitoring System ')

# Input field for category selection
category_input = st.selectbox("Select a Category", ['Fitness', 'Health', 'Nutrition', 'Medicine'])

# Input field for description of user preference
user_input = st.text_input("Describe your need", "e.g., I want to reduce stress")

# Function to get recommendations
def get_recommendations(category, user_input, df):
    # Filter based on category
    filtered_df = df[df['Category'] == category].reset_index(drop=True)

    # Recompute TF-IDF for the filtered data
    filtered_tfidf_matrix = tfidf.transform(filtered_df['Description'])

    # Calculate similarity for the user input
    user_input_vector = tfidf.transform([user_input])
    user_sim = cosine_similarity(user_input_vector, filtered_tfidf_matrix)

    # Get top recommendation (highest similarity score)
    top_idx = user_sim.argsort()[0][-1]  # index of the most similar item
    recommendation = filtered_df.iloc[top_idx]
    
    return recommendation

# Show recommendations when user provides input
if st.button('Get Recommendation'):
    recommendation = get_recommendations(category_input, user_input, df)
    st.write(f"### Recommended: {recommendation['Title']}")
    st.write(f"*Description*: {recommendation['Description']}")
