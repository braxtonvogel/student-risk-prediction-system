# Student Risk Prediction System

A machine learning project that predicts student academic risk using demographic and educational data from the Open University Learning Analytics Dataset (OULAD).

## Overview

This project builds a student risk prediction pipeline capable of:

- Loading and preprocessing educational data
- Training a machine learning classification model
- Predicting whether students are academically at risk
- Evaluating model performance
- Experimenting with explainability techniques (SHAP)
- Simulating intervention strategies to reduce student risk

The project was built as part of a personal machine learning and data science portfolio focused on educational analytics and predictive systems.

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- SHAP
- Jupyter Notebook
- VS Code

---

## Project Structure

```text
Student Risk Personal Project/
│
├── data/
│   └── studentInfo.csv
│
├── src/
│   ├── model.py
│   └── pipeline.py
│
├── notebooks/
│   └── 01_exploration.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Features

### Machine Learning Pipeline
- Data loading
- Data preprocessing
- Train/test splitting
- Random Forest classification
- Model evaluation

### Explainability
- SHAP-based feature contribution analysis
- Student-level prediction interpretation

### Intervention Simulation
- Simulates academic support strategies
- Estimates reduction in student risk probability

---

## Example Output

```python
{
    'risk_score': 0.05,
    'risk_level': 'LOW',
    'best_intervention': 'study_boost',
    'expected_new_risk': 0.01,
    'expected_reduction': 0.04
}
```

---

## Future Improvements

- Streamlit web application
- AI-powered academic support assistant
- Improved feature engineering
- Enhanced intervention modeling
- Real-time dashboard integration

---

## Dataset

Open University Learning Analytics Dataset (OULAD)

https://analyse.kmi.open.ac.uk/open_dataset

---

## Author

Braxton Vogel
Software Engineering Student at Sam Houston State University