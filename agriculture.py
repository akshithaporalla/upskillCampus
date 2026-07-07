import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_excel("dataset/Delinquency_prediction_dataset.xlsx")

print("\nDataset Loaded Successfully\n")

print(df.head())

# -----------------------------
# Remove Missing Values
# -----------------------------

num_cols=[
'Income',
'Credit_Score',
'Loan_Balance'
]

imputer=SimpleImputer(strategy='median')

df[num_cols]=imputer.fit_transform(df[num_cols])

# -----------------------------
# Encode Categorical Variables
# -----------------------------

encoder=LabelEncoder()

categorical_columns=[
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
    df[col]=encoder.fit_transform(df[col].astype(str))

# -----------------------------
# Features
# -----------------------------

X=df.drop(
[
'Customer_ID',
'Delinquent_Account'
],
axis=1
)

y=df['Delinquent_Account']

# -----------------------------
# Split
# -----------------------------

X_train,X_test,y_train,y_test=train_test_split(
X,
y,
test_size=0.20,
random_state=42
)

# -----------------------------
# Model
# -----------------------------

model=LogisticRegression(max_iter=2000)

model.fit(X_train,y_train)

# -----------------------------
# Prediction
# -----------------------------

prediction=model.predict(X_test)

print("\nAccuracy")

print(accuracy_score(y_test,prediction))

print("\nClassification Report")

print(classification_report(y_test,prediction))

# -----------------------------
# Confusion Matrix
# -----------------------------

cm=confusion_matrix(y_test,prediction)

plt.figure(figsize=(5,4))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j,i,str(cm[i][j]),
                 ha="center",
                 va="center",
                 color="black")

plt.colorbar()

plt.savefig("graphs/confusion_matrix.png")

plt.show()

# -----------------------------
# Feature Importance
# -----------------------------

importance=pd.DataFrame({

'Feature':X.columns,

'Coefficient':model.coef_[0]

})

importance=importance.sort_values(
'Coefficient',
ascending=False
)

print("\nFeature Importance\n")

print(importance)

plt.figure(figsize=(10,6))

plt.barh(
importance['Feature'],
importance['Coefficient']
)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("graphs/feature_importance.png")

plt.show()

# -----------------------------
# Predict Entire Dataset
# -----------------------------

df['Prediction']=model.predict(X)

df['Risk_Probability']=model.predict_proba(X)[:,1]

df.to_excel(
"output/Prediction_Output.xlsx",
index=False
)

print("\nProject Executed Successfully")