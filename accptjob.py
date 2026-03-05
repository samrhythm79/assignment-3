import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Job Acceptance Dashboard", layout="wide")

st.title("🎯 Job Acceptance Prediction System")

# ===============================
# LOAD DATA
# ===============================

@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/SARVIN/Downloads/Mini-Project--3/HR_Job_Placement_Dataset.csv")
    return df

df = load_data()

st.subheader("📂 Dataset Overview")
st.write("Shape:", df.shape)
st.dataframe(df.head())


# Load dataset
df = pd.read_csv("C:/Users/SARVIN/Downloads/Mini-Project--3/HR_Job_Placement_Dataset.csv")

# Check basic structure
print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

#step 2 => data understanding
# Shape
print("Shape:", df.shape)

# Data types
print("\nData Types:")
print(df.dtypes)

# Summary statistics
print("\nSummary:")
print(df.describe())

# Null values
print("\nMissing Values:")
print(df.isnull().sum())
#Step 3: Data Cleaning & Preprocessing
#Handling Missing Values
# Numerical columns for initial missing value handling
num_cols_initial = df.select_dtypes(include=np.number).columns
df[num_cols_initial] = df[num_cols_initial].fillna(df[num_cols_initial].median())

# Categorical columns
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing Values After Cleaning:")
print(df.isnull().sum())

#Remove Duplicate Records
df = df.drop_duplicates()
print("Shape After Removing Duplicates:", df.shape)




df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df[col] = df[col].str.strip().str.title()

#Correct Inconsistent Categorical Labels
if "gender" in df.columns:
    df["gender"] = df["gender"].str.strip().str.title()
for col in cat_cols:
    df[col] = df[col].str.strip().str.title()


if "company_tier" in df.columns:
    df["company_tier"] = df["company_tier"].str.replace(" ", "")
if "experience_years" in df.columns:
    df = df[df["experience_years"] >= 0]



if "interview_score" in df.columns:
    df = df[(df["interview_score"] >= 0) & (df["interview_score"] <= 100)]


#ensure Logical Consistency

#Ex:Experience cannot be negative,Scores must be within 0–100

if "experience_years" in df.columns:
    df = df[df["experience_years"] >= 0]

# Assuming `interview_score` is the combined score you calculate later in EDA, if `interview_score` is not yet created here, this check might not apply.
# If `technical_score`, `aptitude_score`, `communication_score` exist, ensure they are within a reasonable range.
# The problem description mentions Interview_Score with a 0-100 range, let's assume those individual scores also fall there if they are being used directly.
if "technical_score" in df.columns:
    df = df[(df["technical_score"] >= 0) & (df["technical_score"] <= 100)]
if "aptitude_score" in df.columns:
    df = df[(df["aptitude_score"] >= 0) & (df["aptitude_score"] <= 100)]
if "communication_score" in df.columns:
    df = df[(df["communication_score"] >= 0) & (df["communication_score"] <= 100)]


#Encode Categorical Variables=> Encode Target Variable
# After column name normalization, the target column will be 'status'
if "status" in df.columns:
    df["status"] = df["status"].map({
        "Placed": 1,
        "Not Placed": 0
    })
#One-Hot Encoding CONVERT CATE TO NUM FORMAT )
df = pd.get_dummies(df, drop_first=True)

# Feature Scaling
scaler = StandardScaler()
# Select only the features to be scaled, explicitly excluding the 'status' target variable.
# 'status' column is expected to be present and numerical (0/1) at this point.
features_to_scale = df.select_dtypes(include=np.number).columns.drop('status', errors='ignore')
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Save cleaned dataset to CSV
df.to_csv("cleaned_job_acceptance_data.csv", index=False)

print("✅ Cleaned JOB ACCEPTANCE DATASET!")


# Step 4: Exploratory Data Analysis (EDA) 


import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (14, 8)


# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)


# -----------------------------
# Plot Functions
# -----------------------------

def plot_target_distribution(df):
    st.subheader("Job Acceptance Distribution")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    counts = df["status"].value_counts()

    axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%",
                colors=["#4CAF50", "#F44336"], startangle=90)
    axes[0].set_title("Placement Status")

    sns.countplot(x="status", data=df, ax=axes[1],
                  palette=["#4CAF50", "#F44336"])
    axes[1].set_title("Count by Status")
    axes[1].bar_label(axes[1].containers[0])

    st.pyplot(fig)


def plot_interview_vs_acceptance(df):
    st.subheader("Interview Score vs Job Acceptance")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    sns.boxplot(x="status", y="interview_score",
                data=df, ax=axes[0],
                palette=["#4CAF50", "#F44336"])
    axes[0].set_title("Interview Score by Status")

    sns.histplot(data=df, x="interview_score",
                 hue="status", bins=30,
                 ax=axes[1],
                 palette=["#4CAF50", "#F44336"], alpha=0.7)
    axes[1].set_title("Interview Score Distribution")

    st.pyplot(fig)


def plot_skills_impact(df):
    st.subheader("Skills Match Impact")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    sns.boxplot(x="status", y="skills_match_percentage",
                data=df, ax=axes[0],
                palette=["#4CAF50", "#F44336"])
    axes[0].set_title("Skills Match % by Status")

    if "skills_level" in df.columns:
        level_rate = df.groupby("skills_level")["status"].mean() * 100
        level_rate = level_rate.reset_index()

        sns.barplot(x="skills_level",
                    y="status",
                    data=level_rate,
                    ax=axes[1],
                    palette="Blues_d")
        axes[1].set_ylabel("Placement Rate (%)")

    st.pyplot(fig)


# -----------------------------
# Main Function
# -----------------------------

def run_eda():

    st.title("Step 4: Exploratory Data Analysis (EDA)")

    df = load_data("C:/Users/SARVIN/OneDrive/Desktop/streamlit/cleaned_job_acceptance_data.csv")

    # Feature Engineering
    df["interview_score"] = (
        df["technical_score"] +
        df["aptitude_score"] +
        df["communication_score"]
    ) / 3

    df["skills_level"] = pd.cut(
        df["skills_match_percentage"],
        bins=[0, 40, 70, 100],
        labels=["Low", "Medium", "High"]
    )

    # Show dataset preview
    st.write("Dataset Preview")
    st.dataframe(df.head())

    # Run Plots
    plot_target_distribution(df)
    plot_interview_vs_acceptance(df)
    plot_skills_impact(df)

    st.success("EDA Completed Successfully!")


if __name__ == "__main__":
    run_eda()



# ==========================================
# Step 5: Feature Engineering 
# ==========================================



st.title("Step 5: Feature Engineering")

# ------------------------------------------
# Load cleaned dataset
# ------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/SARVIN/OneDrive/Desktop/streamlit/cleaned_job_acceptance_data.csv")

df = load_data()

st.write("Original Dataset Preview")
st.dataframe(df.head())


# ------------------------------------------
# 1️⃣ Experience Category
# ------------------------------------------
def categorize_experience(x):
    if x == 0:
        return "Fresher"
    elif x <= 3:
        return "Junior"
    else:
        return "Senior"

df["experience_category"] = df["years_of_experience"].apply(categorize_experience)


# ------------------------------------------
# 2️⃣ Academic Performance Bands
# ------------------------------------------
df["academic_band"] = pd.cut(
    df["degree_percentage"],
    bins=[0, 6.5, 8, 10],
    labels=["Low", "Medium", "High"]
)


# ------------------------------------------
# 3️⃣ Skills Match Level
# ------------------------------------------
df["skills_match_level"] = pd.cut(
    df["skills_match_percentage"],
    bins=[0, 50, 75, 100],
    labels=["Low", "Medium", "High"]
)


# ------------------------------------------
# 4️⃣ Interview Performance Category
# ------------------------------------------
if 'interview_score' not in df.columns:
    df["interview_score"] = (
        df["technical_score"] +
        df["aptitude_score"] +
        df["communication_score"]
    ) / 3

df["interview_category"] = pd.cut(
    df["interview_score"],
    bins=[0, 50, 75, 100],
    labels=["Poor", "Average", "Excellent"]
)


# ------------------------------------------
# 5️⃣ Placement Probability Score
# ------------------------------------------
df["placement_probability_score"] = (
    (df["degree_percentage"] * 0.3) +
    (df["skills_match_percentage"] * 0.3) +
    (df["interview_score"] * 0.3) +
    (df["years_of_experience"] * 2)
)

df["placement_probability_score"] = (
    (df["placement_probability_score"] - df["placement_probability_score"].min()) /
    (df["placement_probability_score"].max() - df["placement_probability_score"].min())
) * 100


# ------------------------------------------
# Save Updated Dataset
# ------------------------------------------
df.to_csv("job_placement_feature_engineered.csv", index=False)



st.write("Updated Dataset Preview")
st.dataframe(df.head())


# ==========================================
# Step 7: Machine Learning Modeling 
# ==========================================


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

st.title("Step 7: Machine Learning Modeling")

# ------------------------------------------
# 1️⃣ Load Feature Engineered Dataset
# ------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("job_placement_feature_engineered.csv")

df = load_data()

st.write("Dataset Preview")
st.dataframe(df.head())


# ------------------------------------------
# 2️⃣ Define Target Variable
# ------------------------------------------
X = df.drop("status", axis=1)
y = df["status"]


# ------------------------------------------
# 4️⃣ Encode Categorical Variables
# ------------------------------------------
categorical_cols = X.select_dtypes(include="object").columns

le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])


# ------------------------------------------
# 5️⃣ Feature Scaling
# ------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ------------------------------------------
# 6️⃣ Train-Test Split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# ------------------------------------------
# 7️⃣ Train Model (Logistic Regression)
# ------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)


# ------------------------------------------
# 8️⃣ Predictions
# ------------------------------------------
y_pred = model.predict(X_test)


# ------------------------------------------
# 9️⃣ Evaluation
# ------------------------------------------
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Evaluation")

st.write("### Model Accuracy")
st.success(f"{accuracy:.4f}")

st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

st.write("### Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))


# ------------------------------------------
# 🔟 Save Model
# ------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/job_acceptance_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

st.success("Model Saved Successfully ✅")



# ==========================================
# 8) Analyst Tasks – EDA & ML Analytics

# ==========================================


import seaborn as sns

sns.set(style="whitegrid")

st.title("Step 8: Analyst Tasks – EDA & ML Analytics")

# ------------------------------------------
# Load Feature Engineered Dataset
# ------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("job_placement_feature_engineered.csv")

df = load_data()

st.write("Dataset Preview")
st.dataframe(df.head())


# ==========================================
# 💠 Candidate Performance Analysis
# ==========================================

st.header("Candidate Performance Analysis")

# 1️⃣ Academic Scores vs Placement
st.subheader("Academic Performance vs Placement")

fig1 = plt.figure(figsize=(6,4))
sns.boxplot(x="status", y="degree_percentage", data=df)
plt.title("Academic Performance vs Placement")
st.pyplot(fig1)


# 2️⃣ Skills Match vs Interview Performance
st.subheader("Skills Match vs Interview Score")

fig2 = plt.figure(figsize=(6,4))
sns.scatterplot(x="skills_match_percentage",
                y="interview_score",
                hue="status",
                data=df)
plt.title("Skills Match vs Interview Score")
st.pyplot(fig2)


# 3️⃣ Certification Impact on Job Acceptance
if "certifications_count" in df.columns:
    st.subheader("Certification Impact on Job Acceptance")

    fig3 = plt.figure(figsize=(6,4))
    sns.countplot(x="certifications_count",
                  hue="status",
                  data=df)
    plt.title("Certification Impact on Job Acceptance")
    st.pyplot(fig3)


# ==========================================
# 💠 Placement & Acceptance Analysis
# ==========================================

st.header("Placement & Acceptance Analysis")

# 4️⃣ Acceptance Rate by Company Tier
if "company_tier" in df.columns:

    tier_acceptance = df.groupby("company_tier")["status"].mean() * 100

    st.write("### Acceptance Rate by Company Tier (%)")
    st.write(tier_acceptance)

    fig4 = plt.figure(figsize=(6,4))
    tier_acceptance.plot(kind="bar")
    plt.title("Acceptance Rate by Company Tier")
    plt.ylabel("Acceptance Rate (%)")
    st.pyplot(fig4)


# 5️⃣ Experience vs Placement Success
st.subheader("Experience vs Placement Success")

fig5 = plt.figure(figsize=(6,4))
sns.boxplot(x="status",
            y="years_of_experience",
            data=df)
plt.title("Experience vs Placement Success")
st.pyplot(fig5)


# ==========================================
# 💠 Interview & Evaluation Analysis
# ==========================================

st.header("Interview & Evaluation Analysis")

# 6️⃣ Interview Score vs Placement Probability
if "placement_probability_score" in df.columns:

    fig6 = plt.figure(figsize=(6,4))
    sns.scatterplot(x="interview_score",
                    y="placement_probability_score",
                    hue="status",
                    data=df)
    plt.title("Interview Score vs Placement Probability")
    st.pyplot(fig6)


st.success("Analyst EDA Completed Successfully ✅")



# =====================================================
# HR KPI Dashboard - Job Acceptance System

# =====================================================



st.set_page_config(page_title="🎯 Job Acceptance Prediction System", layout="wide")

st.title(" 🎯 Job Acceptance Prediction System")

# -----------------------------------------------------
# Load Data
# -----------------------------------------------------
@st.cache_data
def load_data():
    # Keep CSV inside same project folder
    return pd.read_csv("job_placement_feature_engineered.csv")

df = load_data()

# -----------------------------------------------------
# Convert status safely
# -----------------------------------------------------
if "status" in df.columns and df["status"].dtype == "object":
    df["status"] = df["status"].map({"Placed": 1, "Not Placed": 0})

# -----------------------------------------------------
# KPI Calculations
# -----------------------------------------------------

total_candidates = len(df)

placement_rate = df["status"].mean() * 100 if "status" in df.columns else 0

# Offer acceptance
if "offer_accepted" in df.columns:
    job_acceptance_rate = df["offer_accepted"].mean() * 100
else:
    job_acceptance_rate = placement_rate  # fallback

# Interview score safe handling
if "interview_score" in df.columns:
    avg_interview_score = df["interview_score"].mean()
else:
    avg_interview_score = 0

# Skills match
if "skills_match_percentage" in df.columns:
    avg_skills_match = df["skills_match_percentage"].mean()
else:
    avg_skills_match = 0

# Offer Dropout Rate
if "offer_accepted" in df.columns:
    offer_dropout_rate = (1 - df["offer_accepted"].mean()) * 100
else:
    offer_dropout_rate = 0

# High-Risk Candidate %
if "placement_probability_score" in df.columns:
    high_risk_percentage = (
        len(df[df["placement_probability_score"] < 40]) / total_candidates
    ) * 100
else:
    high_risk_percentage = 0


# -----------------------------------------------------
# Display KPIs in Columns
# -----------------------------------------------------

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)
col7, _ = st.columns(2)

col1.metric("Total Candidates", total_candidates)
col2.metric("Placement Rate (%)", f"{placement_rate:.2f}%")
col3.metric("Job Acceptance Rate (%)", f"{job_acceptance_rate:.2f}%")

col4.metric("Average Interview Score", f"{avg_interview_score:.2f}")
col5.metric("Average Skills Match (%)", f"{avg_skills_match:.2f}%")
col6.metric("Offer Dropout Rate (%)", f"{offer_dropout_rate:.2f}%")

col7.metric("High-Risk Candidates (%)", f"{high_risk_percentage:.2f}%")

st.success("KPI Dashboard Loaded Successfully ✅")