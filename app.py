import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Streamlit Page
# -----------------------------
st.set_page_config(page_title="Delinquency Prediction Dashboard", layout="wide")

st.title(" Delinquency Prediction Dashboard")
st.markdown("### Machine Learning Project")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_excel("dataset/Delinquency_prediction_dataset.xlsx")

# -----------------------------
# Missing Value Handling
# -----------------------------
num_cols = [
    'Income',
    'Credit_Score',
    'Loan_Balance'
]

imputer = SimpleImputer(strategy='median')
df[num_cols] = imputer.fit_transform(df[num_cols])

# -----------------------------
# Encode Categorical Columns
# -----------------------------
encoder = LabelEncoder()

categorical_columns = [
    'Employment_Status',
    'Credit_Card_Type',
    'Location',
    'Month_1',
    'Month_2',
    'Month_3',
    'Month_4',
    'Month_5',
    'Month_6'
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col].astype(str))

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop(['Customer_ID', 'Delinquent_Account'], axis=1)
y = df['Delinquent_Account']

# -----------------------------
# Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

# -----------------------------
# Accuracy
# -----------------------------
st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

# -----------------------------
# Classification Report
# -----------------------------
st.subheader("Classification Report")

report = classification_report(
    y_test,
    prediction,
    output_dict=True
)

st.dataframe(pd.DataFrame(report).transpose())

# -----------------------------
# Two Charts
# -----------------------------
col1, col2 = st.columns(2)

# -----------------------------
# Confusion Matrix
# -----------------------------
with col1:

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, prediction)

    fig, ax = plt.subplots(figsize=(5,4))

    cax = ax.matshow(cm, cmap="Blues")

    plt.colorbar(cax)

    ax.set_xlabel("Predicted")

    ax.set_ylabel("Actual")

    ax.set_xticks([0,1])

    ax.set_yticks([0,1])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,str(cm[i,j]),
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=14)

    st.pyplot(fig)

# -----------------------------
# Pie Chart
# -----------------------------
with col2:

    st.subheader("Delinquent vs Non-Delinquent")

    counts = y.value_counts()

    fig2, ax2 = plt.subplots(figsize=(5,4))

    ax2.pie(
        counts,
        labels=["Non-Delinquent","Delinquent"],
        autopct="%1.1f%%",
        startangle=90
    )

    ax2.axis("equal")

    st.pyplot(fig2)

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

importance = importance.sort_values(
    by="Coefficient",
    ascending=False
)

fig3, ax3 = plt.subplots(figsize=(10,6))

ax3.barh(
    importance["Feature"],
    importance["Coefficient"]
)

ax3.set_title("Feature Importance")

plt.tight_layout()

st.pyplot(fig3)

# -----------------------------
# Risk Probability Distribution
# -----------------------------
st.subheader("Risk Probability Distribution")

risk = model.predict_proba(X)[:,1]

fig4, ax2 = plt.subplots(figsize=(4,4))

ax2.hist(
    risk,
    bins=20
)

ax2.set_xlabel("Risk Probability")

ax2.set_ylabel("Number of Customers")

ax2.set_title("Risk Probability Distribution")

st.pyplot(fig4)