import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier,AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    return df

# Function to create features
def create_features(df):
    return df['content'].values

# Function to train the model
def train_model(X, y, data):
    indices = data.index
    indices_train, indices_test = train_test_split(indices, test_size=0.2, random_state=42)

    # Use these indices to select rows for training and testing
    X_train = X[indices_train]
    X_test = X[indices_test]
    y_train = y[indices_train]
    y_test = y[indices_test]

    # Apply SMOTE to handle imbalanced data
    smote = SMOTE(random_state=42)
    
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Scale the features
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    with open('scaler_spam.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    model = LogisticRegression()
    # base_models = [
    #     ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    #     ('xgb', xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,
    #      random_state=42, use_label_encoder=False, eval_metric='logloss')),
    #     ('svc', SVC(kernel='linear', probability=True, random_state=42)),
    #     ('et', ExtraTreesClassifier(n_estimators=100, random_state=42)),
    #     ('knn', KNeighborsClassifier(n_neighbors=5)),
        
    # ]
    # meta_model = LogisticRegression()
    # model = VotingClassifier(estimators=base_models, voting='hard')
    model.fit(X_train_scaled, y_train_resampled)

    # Predictions and evaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Create a DataFrame for results
    results_df = pd.DataFrame({
        'content': data.loc[indices_test, 'content'],
        'real_spamorham': y_test,
        'Predicted_spamorham': y_pred,
    }, index=indices_test)

    return model, accuracy, precision, recall, f1, results_df

# Load and preprocess data
df = load_and_preprocess_data("C:\\Users\\ofekm\\Downloads\\spam.csv")
X = create_features(df)
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)
y = df['spamorham'].apply(lambda x: 1 if x == 'spam' else 0).values

# Save the vectorizer
with open('vectorizer_spam.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

# Train the model and get results
model, accuracy, precision, recall, f1, results_df = train_model(X_transformed, y, df)
with open('spam_detection.pkl', 'wb') as file:
    pickle.dump(model, file)

# Print evaluation metrics
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Save the results DataFrame to a CSV file
results_csv_path = r"C:\work\python\new_predicted_results.csv"
results_df.to_csv(results_csv_path, index=False)

# # Feature importance
# importances = model.feature_importances_
# indices = np.argsort(importances)[::-1]
# feature_names = vectorizer.get_feature_names_out()

# print("Feature importances:")
# for i in range(10):
#     print(f"{feature_names[indices[i]]}: {importances[indices[i]]}")

# Load the saved model, vectorizer, and scaler for predictions on new data
# with open(r'C:\work\python\family_model\scaler_spam.pkl', 'rb') as scaler_file:
#     scaler = pickle.load(scaler_file)
# with open(r'C:\work\python\family_model\spam_detection.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)
# with open(r'C:\work\python\family_model\vectorizer_spam.pkl', 'rb') as vec_file:
#     vectorizer = pickle.load(vec_file)

# # # Prediction on new data
# new_data = ["BANK OF AMERICA: Your transaction of $5,500 will be automatically approved. To deny the transfer or report suspicious activity please click [link] to re-activate your BANK OF AMERICA account and continue using it."]
# features = vectorizer.transform(new_data)
# features_scaled = scaler.transform(features)
# prediction = model.predict(features_scaled)

# if prediction[0] == 0:
#     print("This message is probably not a spam")
# else:
#     print("This message is probably a spam")
