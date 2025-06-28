import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load dataset
df = pd.read_csv('student_performance_sample.csv')

# Step 2: Encode categorical columns
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])  # Male=1, Female=0

le_result = LabelEncoder()
df['Result'] = le_result.fit_transform(df['Result'])  # Pass=1, Fail=0

# Step 3: Define features and target
X = df.drop(columns=['Student_ID', 'Result'])  # input features
y = df['Result']  # target variable

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Make a sample prediction
sample = pd.DataFrame({
    'Attendance (%)': [75],
    'Internal_Marks': [65],
    'Assignments_Submitted': [7],
    'Study_Hours': [2.0],
    'Behavior_Score': [6],
    'Gender': [le_gender.transform(['Female'])[0]]
})
result = model.predict(sample)
print("Predicted Result:", le_result.inverse_transform(result)[0])
