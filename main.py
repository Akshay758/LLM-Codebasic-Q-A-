import streamlit as st

from langchain_helper import create_vector_db, ask_question


# ==========================================
# Dark UI
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #000000;
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

/* Text input */
.stTextInput input {
    background-color: #111111;
    color: white;
    border: 1px solid #444444;
}

/* Input placeholder */
.stTextInput input::placeholder {
    color: #aaaaaa;
}

/* Button */
.stButton button {
    background-color: #222222;
    color: white;
    border: 1px solid #555555;
}

.stButton button:hover {
    background-color: #333333;
    color: white;
}

/* Success message */
.stAlert {
    background-color: #111111;
    color: white;
}

/* Header */
[data-testid="stHeader"] {
    background-color: #000000;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# App
# ==========================================

st.title("AI Knowledge Assistant 🤖")


if st.button("Create Knowledgebase"):

    create_vector_db()

    st.success("Knowledgebase created successfully!")


question = st.text_input("Question:")


if question:

    answer = ask_question(question)

    st.header("Answer")

    st.write(answer)