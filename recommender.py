
import joblib
import pandas as pd

# ============================
# Load Model
# ============================

lgb_model = joblib.load("artifacts/lightgbm_model.pkl")
encoder = joblib.load("artifacts/encoder.pkl")

feature_columns = joblib.load("artifacts/feature_columns.pkl")
categorical_features = joblib.load("artifacts/categorical_features.pkl")

# ============================
# Load Runtime Artifacts
# ============================

student_master = pd.read_pickle("artifacts/student_master.pkl")
resource_master = pd.read_pickle("artifacts/resource_master.pkl")
student_preferences = pd.read_pickle("artifacts/student_preferences.pkl")
feature_importance = pd.read_pickle("artifacts/feature_importance.pkl")

student_course_lookup = joblib.load("artifacts/student_course_lookup.pkl")
student_clicked_lookup = joblib.load("artifacts/student_clicked_lookup.pkl")

# ============================
# Rename Columns
# ============================

student_master = student_master.rename(columns={
    "total_clicks": "student_total_clicks",
    "interaction_count": "student_interaction_count",
    "unique_resources": "student_unique_resources",
    "active_days": "student_active_days",
    "first_day": "student_first_day",
    "last_day": "student_last_day",
    "active_span": "student_active_span",
    "avg_clicks_per_interaction": "student_avg_clicks_per_interaction",
    "avg_clicks_per_day": "student_avg_clicks_per_day",
    "resource_revisit_ratio": "student_resource_revisit_ratio"
})

resource_master = resource_master.rename(columns={
    "total_clicks": "resource_total_clicks",
    "interaction_count": "resource_interaction_count",
    "unique_students": "resource_unique_students",
    "avg_clicks_per_student": "resource_avg_clicks_per_student",
    "avg_clicks_per_interaction": "resource_avg_clicks_per_interaction"
})

# ============================
# Student Profile
# ============================

def get_student_profile(student_id):
    student_features = student_master[
        student_master["id_student"] == student_id
    ]

    student_pref = student_preferences[
        student_preferences["id_student"] == student_id
    ]

    if student_features.empty:
        raise ValueError(f"Student ID {student_id} not found.")

    return student_features.merge(
        student_pref,
        on="id_student",
        how="left"
    )

# ============================
# Candidate Resources
# ============================

def get_candidate_resources(student_id):

    if student_id not in student_course_lookup:
        raise ValueError(f"No course found for student {student_id}.")

    courses = student_course_lookup[student_id]
    code_module, code_presentation = courses[0]


    candidate_resources = resource_master[
        (resource_master["code_module"] == code_module)
        &
        (resource_master["code_presentation"] == code_presentation)
    ].copy()

    clicked_resources = student_clicked_lookup.get(student_id, set())

    candidate_resources = candidate_resources[
        ~candidate_resources["id_site"].isin(clicked_resources)
    ].copy()

    return candidate_resources

# ============================
# Candidate Data
# ============================

def build_candidate_dataframe(student_info, candidate_resources):

    student_info = pd.concat(
        [student_info] * len(candidate_resources),
        ignore_index=True
    )

    candidate_data = pd.concat(
        [
            student_info.reset_index(drop=True),
            candidate_resources.reset_index(drop=True)
        ],
        axis=1
    )

    candidate_data.columns = [
        "id_student","gender","age_band","highest_education","imd_band",
        "studied_credits","num_of_prev_attempts","disability","region",
        "avg_score","median_score","max_score","min_score","score_std",
        "assessment_count","avg_submission_day","first_submission",
        "last_submission","student_total_clicks",
        "student_interaction_count","student_unique_resources",
        "student_active_days","student_first_day","student_last_day",
        "student_active_span","student_avg_clicks_per_interaction",
        "student_avg_clicks_per_day","student_resource_revisit_ratio",
        "pref_forumng","pref_other","pref_oucollaborate",
        "pref_oucontent","pref_page","pref_quiz","pref_resource",
        "pref_subpage","pref_url","id_site","code_module",
        "code_presentation","activity_type","resource_total_clicks",
        "resource_interaction_count","resource_unique_students",
        "resource_avg_clicks_per_student",
        "resource_avg_clicks_per_interaction",
        "resource_revisit_rate"
    ]

    return candidate_data

# ============================
# Model Input
# ============================

def prepare_model_input(candidate_data):

    candidate_X = candidate_data[feature_columns].copy()

    candidate_X[categorical_features] = encoder.transform(
        candidate_X[categorical_features]
    )

    return candidate_X

# ============================
# Prediction
# ============================

def predict_scores(candidate_X):
    return lgb_model.predict_proba(candidate_X)[:, 1]

# ============================
# Recommend
# ============================

def recommend(student_id, top_k=10):

    student_info = get_student_profile(student_id)

    candidate_resources = get_candidate_resources(student_id)

    if candidate_resources.empty:
        return pd.DataFrame()

    candidate_data = build_candidate_dataframe(
        student_info,
        candidate_resources
    )

    candidate_X = prepare_model_input(candidate_data)

    scores = predict_scores(candidate_X)

    recommendations = candidate_data[
        ["id_student","id_site","activity_type"]
    ].copy()

    recommendations["prediction_score"] = scores

    recommendations = (
        recommendations
        .sort_values("prediction_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    recommendations.insert(
        0,
        "Rank",
        range(1, len(recommendations)+1)
    )

    return recommendations

available_student_ids = sorted(
    set(student_master["id_student"])
    &
    set(student_preferences["id_student"])
    &
    set(student_course_lookup.keys())
)

if __name__ == "__main__":
    print(recommend(8462,10))
