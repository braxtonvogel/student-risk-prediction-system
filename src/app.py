import streamlit as st
import pandas as pd
import ollama
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# =========================
# PAGE
# =========================
st.set_page_config(page_title="Student Risk AI System")
st.title("🎓 Student Risk Prediction System")


# =========================
# STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# CONTEXT
# =========================
def get_context(df, df_results):
    if df_results is not None:
        return df_results.head(5).to_string(index=False)
    if df is not None:
        return df.head(5).to_string(index=False)
    return "No dataset loaded yet."


# =========================
# AI (UNCHANGED - AS REQUESTED)
# =========================
def ask_ai(question, context):
    try:
        response = ollama.chat(
            model="llama3:latest",
            messages=[
                {"role": "system", "content": "You are Nova, an AI assistant for student risk analysis."},
                {"role": "user", "content": f"{context}\n\n{question}"}
            ]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"AI Error: {e}"


# =========================
# DATA
# =========================
data_option = st.radio(
    "Choose data source:",
    [
        "Upload CSV",
        "Use Default Dataset (Open University Learning Analytics Dataset - OULAD)"
    ]
)

df = None
df_results = None

if data_option.startswith("Upload"):
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/studentInfo.csv")
    st.success("Using OULAD (Open University Learning Analytics Dataset)")


# =========================
# MODEL
# =========================
model = None
df_ml = None

if df is not None and "final_result" in df.columns:

    df_ml = df.copy()
    df_ml["risk"] = df_ml["final_result"].apply(
        lambda x: 1 if x in ["Fail", "Withdrawn"] else 0
    )

    X = df_ml.select_dtypes(include=["number"]).fillna(0)
    y = df_ml["risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)


# =========================
# PREDICTIONS
# =========================
if model is not None:

    st.markdown("### 🔮 AI-Powered Risk Predictions")

    X_all = df_ml.select_dtypes(include=["number"]).fillna(0)

    preds = model.predict(X_all)
    probs = model.predict_proba(X_all)[:, 1]

    df_results = df.copy()
    df_results["Risk_Level"] = preds
    df_results["Risk_Probability"] = probs

    st.dataframe(df_results)


    # =========================================================
    # 🧠 STUDENT INTERVENTION LAYER (RESTORED + IMPROVED)
    # =========================================================
    st.markdown("### 📘 What Students Can Do to Improve Their Chances")

    def get_actions(risk):
        if risk > 0.75:
            return [
                "Mandatory tutoring sessions",
                "Academic advisor intervention",
                "Weekly progress tracking"
            ]
        elif risk > 0.4:
            return [
                "Join study groups",
                "Attend office hours",
                "Improve assignment consistency"
            ]
        else:
            return [
                "Maintain current performance",
                "Light revision support",
                "Peer discussion groups"
            ]

    def success_probability(risk):
        if risk > 0.75:
            return "High impact if applied (70–85% improvement likelihood)"
        elif risk > 0.4:
            return "Moderate impact (50–70% improvement likelihood)"
        else:
            return "Low impact needed (already performing well)"

    for i, row in df_results.head(10).iterrows():
        st.write(f"#### Student {i}")

        st.write(f"Risk Probability: **{row['Risk_Probability']:.2f}**")

        for action in get_actions(row["Risk_Probability"]):
            st.write(f"- {action}")

        st.info(success_probability(row["Risk_Probability"]))


# =========================
# SIDEBAR (AI SECTION - UNTOUCHED)
# =========================
st.sidebar.title("🤖 Nova AI Assistant")

thinking_box = st.sidebar.empty()

user_input = st.sidebar.chat_input(
    "Ask Nova anything... Note: First request may take a moment"
)


# =========================
# PROCESS INPUT + THINKING ANIMATION
# =========================
if user_input:

    context = get_context(df, df_results)

    st.session_state.messages.append(("user", user_input))

    stages = [
        "🤖 Nova is reading your question...",
        "📊 Analyzing data context...",
        "🧠 Generating insights...",
        "✨ Finalizing response..."
    ]

    for stage in stages:
        thinking_box.info(stage)
        time.sleep(0.25)

    answer = ask_ai(user_input, context)

    thinking_box.empty()

    st.session_state.messages.append(("assistant", answer))


# =========================
# CHAT DISPLAY (NEWEST FIRST)
# =========================
for role, msg in reversed(st.session_state.messages):
    st.sidebar.chat_message(role).write(msg)