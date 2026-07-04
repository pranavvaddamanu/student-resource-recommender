
import streamlit as st
import pandas as pd
import plotly.express as px

from recommender import (
    recommend,
    get_student_profile,
    feature_importance,
    available_student_ids
)

st.set_page_config(page_title="Student Resource Recommendation System",
                   page_icon="🎓",
                   layout="wide")

st.markdown("""
<style>
.main{background:linear-gradient(180deg,#eef4ff,#f9fbff);}
.block-container{padding-top:1.5rem;}
.hero{background:linear-gradient(90deg,#2563eb,#7c3aed);padding:1.4rem;border-radius:16px;color:white;margin-bottom:1rem;}
[data-testid="stMetric"]{background:white;border-radius:12px;padding:10px;border:1px solid #ddd;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🎓 Student Resource Recommendation System</h1>
<p>AI-powered personalized learning resource recommendations using LightGBM.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("⚙️ Settings")

student_id = st.sidebar.selectbox(
    "Select Student",
    available_student_ids,
    index=available_student_ids.index(6516)
)

top_k = st.sidebar.slider("Top Recommendations",5,20,10)

if not st.sidebar.button("🚀 Generate Recommendations", use_container_width=True):
    st.info("Select a student and click Generate Recommendations.")
    st.stop()

profile = get_student_profile(student_id).iloc[0]

left,right = st.columns([1,2])

with left:
    st.subheader("📊 Student Profile")
    a,b = st.columns(2)
    with a:
        st.metric("Gender",profile["gender"])
        st.metric("Education",profile["highest_education"])
        st.metric("Average Score",round(profile["avg_score"],2))
    with b:
        st.metric("Region",profile["region"])
        st.metric("Credits",int(profile["studied_credits"]))
        st.metric("Assessments",int(profile["assessment_count"]))

with right:
    st.subheader("📈 Learning Preferences")
    pref = pd.DataFrame({
        "Resource Type":["Forum","Other","OU Collaborate","OU Content","Page","Quiz","Resource","Subpage","URL"],
        "Preference Score":[
            profile["pref_forumng"],
            profile["pref_other"],
            profile["pref_oucollaborate"],
            profile["pref_oucontent"],
            profile["pref_page"],
            profile["pref_quiz"],
            profile["pref_resource"],
            profile["pref_subpage"],
            profile["pref_url"]
        ]
    })
    fig=px.bar(pref,x="Preference Score",y="Resource Type",orientation="h",
               color="Preference Score",color_continuous_scale="Blues")
    fig.update_layout(template="plotly_white",height=420,coloraxis_showscale=False)
    st.plotly_chart(fig,use_container_width=True)

st.divider()

st.subheader("🎯 Top Recommendations")
rec = recommend(student_id,top_k)
rec["Recommendation Confidence (%)"]=(rec["prediction_score"]*100).round(2)
display = rec.rename(columns={
    "id_site":"Resource ID",
    "activity_type":"Resource Type"
})[["Rank","Resource ID","Resource Type","Recommendation Confidence (%)"]]
st.dataframe(display,hide_index=True,use_container_width=True)

st.markdown("#### Confidence")
for _, r in display.iterrows():
    st.write(f"**#{int(r['Rank'])}** • Resource **{int(r['Resource ID'])}** ({r['Resource Type']})")
    st.progress(float(r["Recommendation Confidence (%)"])/100)
    st.caption(f"{r['Recommendation Confidence (%)']}%")

st.divider()

left,right = st.columns([2,1])

with left:
    st.subheader("📊 Feature Importance")
    imp = feature_importance.head(15)
    fig2 = px.bar(imp,x="Importance (%)",y="Feature",orientation="h",
                  color="Importance (%)",color_continuous_scale="Purples")
    fig2.update_layout(template="plotly_white",height=500,coloraxis_showscale=False)
    st.plotly_chart(fig2,use_container_width=True)

with right:
    st.subheader("📈 Model Performance")
    a,b = st.columns(2)
    with a:
        st.metric("Accuracy","84.80%")
        st.metric("Recall","86.29%")
        st.metric("ROC-AUC","92.93%")
        st.metric("nDCG@10","97.26%")
    with b:
        st.metric("Precision","84.08%")
        st.metric("F1 Score","85.17%")
        st.metric("Hit Rate@10","98.34%")

st.divider()
st.caption("Built with Streamlit • LightGBM • OULAD Dataset")
