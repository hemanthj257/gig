from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import joblib
import numpy as np

def train_diabetes_model():
    # Read data
    path = "/home/hehe/gig/diabetes_prediction_dataset.csv"
    data = pd.read_csv(path)

    # Preprocess categorical variables
    le_gender = LabelEncoder()
    le_smoking = LabelEncoder()
    
    data['gender'] = le_gender.fit_transform(data['gender'])
    data['smoking_history'] = le_smoking.fit_transform(data['smoking_history'])

    # Prepare features and target
    X = data[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
              'bmi', 'HbA1c_level', 'blood_glucose_level']]
    y = data['diabetes']

    # Split and scale data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Save model and scaler
    joblib.dump(model, 'diabetes_model.joblib')
    joblib.dump(scaler, 'diabetes_scaler.joblib')
    joblib.dump(le_gender, 'gender_encoder.joblib')
    joblib.dump(le_smoking, 'smoking_encoder.joblib')

    return model, scaler, le_gender, le_smoking

def predict_diabetes(gender, age, hypertension, heart_disease, smoking, bmi, hba1c, glucose):
    try:
        # Load saved models
        model = joblib.load('diabetes_model.joblib')
        scaler = joblib.load('diabetes_scaler.joblib')
        le_gender = joblib.load('gender_encoder.joblib')
        le_smoking = joblib.load('smoking_encoder.joblib')
    except:
        # Train if models don't exist
        model, scaler, le_gender, le_smoking = train_diabetes_model()

    # Transform inputs
    try:
        gender_encoded = le_gender.transform([str(gender)])[0]
        smoking_encoded = le_smoking.transform([str(smoking)])[0]
    except:
        # Handle numeric inputs
        gender_encoded = int(gender)
        smoking_encoded = int(smoking)

    # Create feature array
    features = np.array([[
        gender_encoded,
        float(age),
        int(hypertension),
        int(heart_disease),
        smoking_encoded,
        float(bmi),
        float(hba1c),
        float(glucose)
    ]])

    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    return prediction, probability

if __name__ == '__main__':
    # Train model if running directly
    train_diabetes_model()


