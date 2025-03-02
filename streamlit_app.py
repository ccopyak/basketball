import streamlit as st
import pandas as pd
import joblib

# Load trained Logistic Regression model
model = joblib.load("log_reg_model.pkl")

# Load team dataset
df = pd.read_excel("data.xlsx")

# Streamlit UI
st.title("🏀 Basketball Game Predictor")

# Select teams
team_names = df["Team"].dropna().unique().tolist()

home_team = st.selectbox("Select Home Team", team_names)
away_team = st.selectbox("Select Away Team", team_names)

if st.button("Predict Winner"):
    # Extract team stats
    home_team_info = df[df["Team"] == home_team]
    away_team_info = df[df["Team"] == away_team]

    if home_team_info.empty or away_team_info.empty:
        st.error("Team data not found!")
    else:
        # Prepare input features
        home_team_data = home_team_info[['NetRtg', 'ORtg', 'DRtg', 'AdjT', 'Luck']].values.flatten()
        away_team_data = away_team_info[['NetRtg', 'ORtg', 'DRtg', 'AdjT', 'Luck']].values.flatten()

        # Construct input features for prediction
        features = {
                'Home NetRtg': home_team_data[0],
                'Home Ortg': home_team_data[1],
                'Home DRtg': home_team_data[2],
                'Home AdjT': home_team_data[3],
                'Home Luck': home_team_data[4],
                'Away NetRtg': away_team_data[0],
                'Away Ortg': away_team_data[1],
                'Away DRtg': away_team_data[2],
                'Away AdjT': away_team_data[3],
                'Away Luck': away_team_data[4]
            }

        features_df = pd.DataFrame([features])
        
        print(type(model))  # See what type of object model is
        print(dir(model))  # See available methods

        # Make prediction
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[:, 1][0]

        # Show result
        if prediction == 1:
            st.success(f"🏆 **Prediction: {home_team} Wins!**")
            st.info(f"Win Probability: {probability * 100:.1f}%")
        else:
            st.success(f"🏆 **Prediction: {away_team} Wins!**")
            st.info(f"Win Probability: {(1 - probability) * 100:.1f}%")