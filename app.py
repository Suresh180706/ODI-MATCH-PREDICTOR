import streamlit as st
import joblib
import pandas as pd

# -------------------------------------------------
# LOAD DATA (for known teams & venues)
# -------------------------------------------------
match_info = pd.read_csv("ODI_Match_info.csv")

known_teams = set(match_info['team1']).union(set(match_info['team2']))
known_venues = sorted(match_info['venue'].dropna().unique())

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="ODI Match Predictor", layout="centered")

# -------------------------------------------------
# LOAD TRAINED MODEL
# -------------------------------------------------
model = joblib.load("model.pkl")

st.title("🏏 ODI Match Winner Prediction App")
st.write("Pre-Match ODI Winner Prediction using Machine Learning")

# -------------------------------------------------
# USER INPUTS
# -------------------------------------------------
st.header("📋 Enter Match Details")

team1 = st.text_input("Team 1", "India")
team2 = st.text_input("Team 2", "Australia")

venue = st.selectbox("Venue", known_venues)
toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.selectbox("Toss Decision", ["bat", "field"])
season_type = st.selectbox("Season", ["Summer", "Rainy", "Winter"])

# -------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------
if st.button("🔮 Predict Winner"):

    # 1️⃣ VALIDATION
    if team1.strip().lower() == team2.strip().lower():
        st.error("❌ Team 1 and Team 2 cannot be the same.")
        st.stop()

    team1_known = team1 in known_teams
    team2_known = team2 in known_teams

    # -------------------------------------------------
    # CASE 1: BOTH TEAMS KNOWN → ML (CORRECT)
    # -------------------------------------------------
    if team1_known and team2_known:

        input_data = pd.DataFrame({
            'team1': [team1],
            'team2': [team2],
            'venue': [venue],
            'toss_winner': [toss_winner],
            'toss_decision': [toss_decision],
            'season_type': [season_type],
            'dl_applied': [0],
            'team1_strength': [0.5],
            'team2_strength': [0.5],
            'team1_bat_strength': [1.0],
            'team2_bat_strength': [1.0],
            'team1_bowl_strength': [1.0],
            'team2_bowl_strength': [1.0],
            'venue_strength': [0.5]
        })

        # Predict probabilities
        proba = model.predict_proba(input_data)[0]
        classes = model.classes_

        # Restrict to Team1 & Team2 only
        team_probs = {}
        for team, p in zip(classes, proba):
            if team == team1 or team == team2:
                team_probs[team] = p

        predicted_winner = max(team_probs, key=team_probs.get)
        probability = team_probs[predicted_winner]

        st.success(f"🏆 Predicted Winner (ML): **{predicted_winner}**")
        st.info(f"Winning Probability: **{round(probability * 100, 2)}%**")

    # -------------------------------------------------
    # CASE 2: ONE KNOWN, ONE UNKNOWN → KNOWN TEAM
    # -------------------------------------------------
    elif team1_known and not team2_known:
        st.success(f"🏆 Predicted Winner: **{team1}**")
        st.info("Reason: Team 1 is known, Team 2 is new")

    elif team2_known and not team1_known:
        st.success(f"🏆 Predicted Winner: **{team2}**")
        st.info("Reason: Team 2 is known, Team 1 is new")

    # -------------------------------------------------
    # CASE 3: BOTH TEAMS UNKNOWN → RULE-BASED
    # -------------------------------------------------
    else:
        if toss_decision == "bat":
            predicted_winner = toss_winner
        else:
            predicted_winner = team1 if toss_winner == team2 else team2

        st.success(f"🏆 Predicted Winner (Rule-Based): **{predicted_winner}**")
        st.info("Reason: Both teams are new → using toss & venue logic")
