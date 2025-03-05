import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import accuracy_score #Added import

# load dataset
df = pd.read_csv('Loan-Approval-Prediction.csv')
print(f"DataFrame shape: {df.shape}")

# fill the missing data
categorical_data = ["Gender", "Married", "Self_Employed", "Dependents"]
for col in categorical_data:
    df[col] = df[col].fillna(df[col].mode()[0])

df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].mean())
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
df["CoapplicantIncome"] = df["CoapplicantIncome"].fillna(df["CoapplicantIncome"].mode()[0])
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
df.drop_duplicates(inplace=True)

x = df.drop("Loan_Status", axis=1)
x = x.drop("Loan_ID", axis=1)

encoder_x = LabelEncoder()
x["Education"] = encoder_x.fit_transform(x["Education"])
x["Married"] = encoder_x.fit_transform(x["Married"])
x["Property_Area"] = encoder_x.fit_transform(x["Property_Area"])
x["Gender"] = encoder_x.fit_transform(x["Gender"])
x["Self_Employed"] = encoder_x.fit_transform(x["Self_Employed"])
x["Dependents"] = encoder_x.fit_transform(x["Dependents"])

encoder_y = LabelEncoder()
y = encoder_y.fit_transform(df["Loan_Status"])

x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"x_train shape: {x_train.shape}, x_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

model = RandomForestClassifier()
print("Training model...")
model.fit(x_train, y_train)
print("Model training complete.")

joblib.dump(model, "l5sod.pkl")
print("Model saved as l5sod.pkl")

# Make predictions and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
