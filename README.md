# 🎓 Student Resource Recommendation System

An end-to-end machine learning recommendation system that generates personalized learning resource recommendations for students using the Open University Learning Analytics Dataset (OULAD). The project combines feature engineering, a LightGBM classification model, and an interactive Streamlit dashboard to recommend unseen learning resources based on student characteristics, historical learning behaviour, and resource attributes.
URL: https://student-resource-recommender-pjuwgpka6cnfkzwwzgdezz.streamlit.app/
---

## 📌 Project Overview

Traditional learning platforms often present the same learning materials to every student. This project demonstrates how machine learning can be used to personalize learning experiences by predicting which learning resources a student is most likely to interact with next.

Given a student's academic profile, historical interactions, and learning preferences, the system recommends the Top-K unseen resources with the highest predicted interaction probability.

---

## 🚀 Features

* Personalized Top-K resource recommendations
* LightGBM-based recommendation model
* Student demographic and assessment feature engineering
* Behaviour-based learning preference modelling
* Resource popularity and engagement features
* Interactive Streamlit dashboard
* Recommendation confidence scores
* Feature importance visualization
* Student learning preference visualization
* Optimized lookup-based recommendation engine for faster inference

---

## 📊 Dataset

This project uses the **Open University Learning Analytics Dataset (OULAD)**.

The dataset contains anonymized information about students, assessments, and interactions with learning resources within the Open University's Virtual Learning Environment (VLE).

Data sources used include:

* Student information
* Student assessments
* Student VLE interactions
* VLE resource metadata

---

## 🧠 Feature Engineering

Several groups of features were created before model training.

### Student Features

* Gender
* Age band
* Highest education
* Region
* Studied credits
* Number of previous attempts
* Disability status

### Assessment Features

* Average score
* Median score
* Maximum score
* Minimum score
* Score standard deviation
* Assessment count
* Average submission day
* First submission
* Last submission

### Student Interaction Features

* Total clicks
* Interaction count
* Unique resources visited
* Active days
* Active span
* Average clicks per interaction
* Average clicks per day
* Resource revisit ratio

### Student Learning Preferences

Historical interactions were converted into preference scores representing the proportion of interactions with different resource types.

Examples include:

* OU Content
* Resource
* URL
* Forum
* Quiz
* Subpage
* Other

### Resource Features

* Activity type
* Total clicks
* Interaction count
* Unique students
* Average clicks per student
* Average clicks per interaction
* Resource revisit rate

---

## 🤖 Machine Learning Model

The recommendation engine uses a **LightGBM classifier**.

For every unseen resource within a student's enrolled course, the model predicts the probability that the student will interact with that resource.

Resources are ranked according to these predicted probabilities, and the Top-K resources are returned as recommendations.

---

## ⚙️ Recommendation Pipeline

1. Select a student.
2. Retrieve the student's demographic, assessment, interaction, and preference features.
3. Retrieve all unseen resources from the student's enrolled course.
4. Create student-resource candidate pairs.
5. Apply the trained encoder to categorical features.
6. Generate interaction probabilities using the LightGBM model.
7. Rank resources by predicted probability.
8. Display the Top-K recommendations.

---

## 📈 Model Performance

| Metric      |  Score |
| ----------- | -----: |
| Accuracy    | 84.80% |
| Precision   | 84.08% |
| Recall      | 86.29% |
| F1 Score    | 85.17% |
| ROC-AUC     | 92.93% |
| Hit Rate@10 | 98.34% |
| nDCG@10     | 97.26% |

---

## 🖥️ Streamlit Dashboard

The application includes:

* Student selection
* Student profile summary
* Learning preference visualization
* Personalized Top-K recommendations
* Recommendation confidence scores
* Feature importance visualization
* Model performance summary

---

## 📂 Project Structure

```text
Student-Resource-Recommender/
│
├── app.py
├── recommender.py
├── requirements.txt
├── README.md
│
├── lightgbm_model.pkl
├── encoder.pkl
├── feature_columns.pkl
├── categorical_features.pkl
├── feature_importance.pkl
│
├── student_master.pkl
├── student_preferences.pkl
├── resource_master.pkl
├── student_course_lookup.pkl
├── student_clicked_lookup.pkl
```

---

## ▶️ Running the Project

Clone the repository:

```bash
git clone <repository-url>
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🔮 Future Improvements

* Temporal or leave-one-out evaluation for recommendation metrics
* Cold-start recommendation support for new students
* Hybrid recommendation models combining collaborative and content-based approaches
* Explainable recommendations using SHAP values
* User authentication and persistent recommendation history

---
SCREENSHOTS:
<img width="1912" height="863" alt="image" src="https://github.com/user-attachments/assets/e7301439-9dcf-4ee7-8acf-8c01659b4f38" />
<img width="1877" height="680" alt="image" src="https://github.com/user-attachments/assets/99ba9510-389e-40e7-b1e1-b175a82fb5f1" />
<img width="1917" height="862" alt="image" src="https://github.com/user-attachments/assets/e246a624-adbc-47aa-892a-32a6e1d99a48" />
<img width="1907" height="847" alt="image" src="https://github.com/user-attachments/assets/fbe83250-65f1-4dcf-a6f9-803d7c1ccd67" />




---

## 👨‍💻 Author

**Pranav Vaddamanu**

Built using **Python**, **Pandas**, **Scikit-learn**, **LightGBM**, **Plotly**, and **Streamlit**.
