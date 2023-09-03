# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("A web-based artificial intelligence (AI) application for predicting persistent critical illness (PerCI) in orthopaedic trauma patients: a prospective cohort study using machine learning")
st.sidebar.title("Selection of Parameters")
st.sidebar.markdown("Picking up parameters")

Pelvic_fractrue = st.sidebar.selectbox("Pelvic fractrue", ("No", "Closed", "Open"))
Ventilation = st.sidebar.selectbox("Mechanical ventilation", ("No", "Yes"))
Respiratory_failure = st.sidebar.selectbox("Respiratory failure", ("No", "Yes"))
Pneumonia = st.sidebar.selectbox("Pneumonia", ("No", "Yes"))
Sepsis = st.sidebar.selectbox("Sepsis", ("No", "Yes"))
Heart_rate = st.sidebar.slider("Heart rate (Beats per minute)", 60, 120)
Respiratory_rate = st.sidebar.slider("Respiratory rate (Beats per minute)", 12, 30)
Albumin = st.sidebar.slider("Albumin (g/dL)", 2.00, 5.00)
Glucose = st.sidebar.slider("Glucose (mg/dL)", 90.0, 190.0)
Calcium = st.sidebar.slider("Total serum calcium (mg/dL)", 6.00, 10.00)
Hematocrit = st.sidebar.slider("Hematocrit (%)", 25.00, 45.00)
OASIS = st.sidebar.slider("OASIS",  20, 45)
SAPSII = st.sidebar.slider("SAPSII", 20, 50)
SOFA = st.sidebar.slider("SOFA", 0, 12)

if st.button("Submit"):
    Xgbc_clf = jl.load("Xgbc_clf_final_round.pkl")
    x = pd.DataFrame([[Pelvic_fractrue, Ventilation, Respiratory_failure, Pneumonia, Sepsis, Heart_rate, Respiratory_rate, Albumin, Glucose, Calcium, Hematocrit, OASIS,
                              SAPSII, SOFA]],
                     columns=["Pelvic_fractrue", "Ventilation", "Respiratory_failure", "Pneumonia", "Sepsis", "Heart_rate", "Respiratory_rate", "Albumin", "Glucose", "Calcium", "Hematocrit", "OASIS",
                              "SAPSII", "SOFA"])

    x = x.replace(["No", "Yes"], [0, 1])
    x = x.replace(["No", "Closed", "Open"], [0, 1, 2])

    # Get prediction
    prediction = Xgbc_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.text(f"Probability of developing PerCI: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.586:
        st.text(f"Risk group: low-risk group")
    else:
        st.text(f"Risk group: High-risk group")
    if prediction < 0.586:
        st.markdown(f"Low-risk group measures: Patients identified as low risk for PerCI require a focus on maintaining their stable condition and preventing potential deterioration. Regular monitoring and assessments should still be conducted to detect any early signs of PerCI. However, the intensity of interventions and monitoring may be adjusted based on the patient's low-risk status. A conservative approach can be taken, ensuring stability, minimizing unnecessary procedures, and avoiding potential risks associated with aggressive management. Patient education on self-management techniques and awareness of signs requiring medical assistance can empower them to actively participate in their care and prevent complications. It is important to consider that while the AI application provides risk estimates and recommendations, clinical decision-making should also incorporate healthcare providers' expertise and individual patient context.")
    else:
        st.markdown(f"High-risk group measures: For patients identified as high risk for persistent critical illness (PerCI), proactive measures should be implemented to prevent the development and progression of this condition. Close monitoring and frequent assessments are recommended, including vital signs, laboratory values, and clinical indicators associated with PerCI. Early intervention strategies such as optimizing fluid balance, ensuring adequate pain control, and early mobilization should be employed to mitigate the risk of PerCI. Prompt identification and management of comorbidities or complications are crucial in this group. It is important to consider that while the AI application provides risk estimates and recommendations, clinical decision-making should also incorporate healthcare providers' expertise and individual patient context.")
st.subheader('Model information')
st.markdown('The eXGBM model demonstrated excellent prediction performance, with the highest value of AUC (0.979, 95%CI: 0.970-0.991) among all the models evaluated. It outperformed the RF model (AUC: 0.957, 95%CI: 0.941-0.967) and the SVM model (AUC: 0.911, 95%CI: 0.878-0.928). The LR model had a relatively lower AUC of 0.753 (95%CI: 0.714-0.793).In terms of accuracy, precision, recall, specificity, F1 score, Brier score, and Log loss, the eXGBM model performed the best among all the models evaluated. External validation of the eXGBM model demonstrated a high AUC of 0.887 (95%CI: 0.863-0.917), indicating good generalizability.')
