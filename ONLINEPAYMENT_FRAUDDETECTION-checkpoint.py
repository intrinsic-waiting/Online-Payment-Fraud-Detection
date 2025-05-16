#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
file_path = "C:/Users/khira/OneDrive/Desktop/online_payment_fraud_detection.csv"
data = pd.read_csv(file_path)
data


# In[2]:


data.head()


# In[3]:


data.tail()


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[5]:


# Drop any irrelevant columns
data = data.drop(columns=['Unnamed: 0', 'Transaction Date', 'User ID'])


# In[6]:


data


# In[7]:


# Convert categorical variables to numerical using Label Encoding
label_encoders = {}
categorical_cols = ['Payment Method', 'Transaction Location', 'Device Type', 'Previous Fraudulent Activity']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store encoders for each column


# In[8]:


label_encoders 


# In[9]:


categorical_cols


# In[10]:


# Separate features and target
X = data.drop(columns=['Is Fraud'])
y = data['Is Fraud']


# In[11]:


X
y


# In[12]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test


# In[13]:


# Initialize scalers for numerical data normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
scaler


# In[14]:


X_train_scaled
X_test_scaled


# In[15]:


# Models: SVM, Decision Tree, and Pipeline with SVM
svm_model = SVC()
decision_tree_model = DecisionTreeClassifier()


# In[16]:


svm_model


# In[17]:


decision_tree_model


# In[18]:


# Pipeline for SVM with scaling
pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])


# In[19]:


pipeline_svm 


# In[20]:


# Train the models
svm_model.fit(X_train_scaled, y_train)
decision_tree_model.fit(X_train, y_train)
pipeline_svm.fit(X_train, y_train)


# In[21]:


# Predictions and accuracy scores
svm_pred = svm_model.predict(X_test_scaled)
decision_tree_pred = decision_tree_model.predict(X_test)
pipeline_svm_pred = pipeline_svm.predict(X_test)


# In[22]:


svm_pred


# In[23]:


decision_tree_pred
pipeline_svm_pred


# In[24]:


# Calculate accuracy for each model
svm_accuracy = accuracy_score(y_test, svm_pred)
decision_tree_accuracy = accuracy_score(y_test, decision_tree_pred)
pipeline_svm_accuracy = accuracy_score(y_test, pipeline_svm_pred)

svm_accuracy, decision_tree_accuracy, pipeline_svm_accuracy


# In[25]:


# Calculate accuracy for each model
svm_accuracy = accuracy_score(y_test, svm_pred)
decision_tree_accuracy = accuracy_score(y_test, decision_tree_pred)
pipeline_svm_accuracy = accuracy_score(y_test, pipeline_svm_pred)

# Print accuracy results for comparison
print("Model Accuracy Comparison:")
print(f"SVM Model Accuracy: {svm_accuracy * 100:.2f}%")
print(f"Decision Tree Model Accuracy: {decision_tree_accuracy * 100:.2f}%")
print(f"Pipeline with SVM Model Accuracy: {pipeline_svm_accuracy * 100:.2f}%")


# In[26]:


import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns



# In[27]:


# Set plot style
sns.set(style="whitegrid")



# In[28]:


# Corrected Decision Tree Visualization Code
plt.figure(figsize=(15, 10))
tree.plot_tree(decision_tree_model, filled=True, feature_names=list(X.columns), class_names=["Not Fraud", "Fraud"], rounded=True)
plt.title("Decision Tree Structure")
plt.show()



# In[29]:


accuracies = {'SVM': svm_accuracy, 'Decision Tree': decision_tree_accuracy, 'Pipeline (SVM)': pipeline_svm_accuracy}
plt.figure(figsize=(8, 6))
plt.bar(accuracies.keys(), accuracies.values(), color=['skyblue', 'salmon', 'limegreen'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.0)
plt.show()


# In[30]:


class_counts = y.value_counts()
plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=['Not Fraud', 'Fraud'], autopct='%1.1f%%', startangle=140, colors=['lightcoral', 'lightskyblue'])
plt.title("Class Distribution (Fraud vs. Not Fraud)")
plt.show()


# In[31]:


#FeatureDistributionHistogram
numeric_features = ['Transaction Amount', 'Time Between Transactions (min)']
data[numeric_features].hist(bins=20, figsize=(12, 6), color='c')
plt.suptitle("Feature Distribution Histograms")
plt.show()


# In[32]:


from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np



# In[33]:


# AUC-ROC Curve for Decision Tree and SVM Models
# Calculate predicted probabilities for ROC curves
svm_probs = svm_model.decision_function(X_test_scaled)
decision_tree_probs = decision_tree_model.predict_proba(X_test)[:, 1]



# In[34]:


# Compute ROC AUC scores
svm_auc = roc_auc_score(y_test, svm_probs)
decision_tree_auc = roc_auc_score(y_test, decision_tree_probs)



# In[35]:


# Plot ROC curves
plt.figure(figsize=(10, 6))
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probs)
fpr_tree, tpr_tree, _ = roc_curve(y_test, decision_tree_probs)
plt.plot(fpr_svm, tpr_svm, color='blue', label=f'SVM (AUC = {svm_auc:.2f})')
plt.plot(fpr_tree, tpr_tree, color='red', label=f'Decision Tree (AUC = {decision_tree_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[36]:


# Outlier Detection

def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers


# In[37]:


# Detect outliers in numerical columns
outliers_amount = detect_outliers(data, 'Transaction Amount')
outliers_time = detect_outliers(data, 'Time Between Transactions (min)')



# In[38]:


outliers_info = {
    'Transaction Amount Outliers': len(outliers_amount),
    'Time Between Transactions Outliers': len(outliers_time)
}
outliers_info


# In[39]:


feature_stds = X.std()
feature_stds



# In[40]:


normalized_data = scaler.transform(X)
normalized_df = pd.DataFrame(normalized_data, columns=X.columns)



# In[41]:


normalized_df.head()


# In[48]:


import numpy as np
import pandas as pd

def get_user_input():
    # Prompt user for each input feature, stripping extra spaces or tabs
    transaction_amount = float(input("Enter Transaction Amount: ").strip())
    payment_method = input("Enter Payment Method (e.g., PayPal, Bank Transfer, Debit Card): ").strip()
    transaction_location = input("Enter Transaction Location (e.g., UK, Canada, France): ").strip()
    device_type = input("Enter Device Type (e.g., Mobile, Tablet, Desktop): ").strip()
    previous_fraud = input("Previous Fraudulent Activity? (Yes/No): ").strip()
    time_between_transactions = float(input("Enter Time Between Transactions (min): ").strip())
    
    # Encode categorical inputs using previously fitted LabelEncoders
    payment_method_encoded = label_encoders['Payment Method'].transform([payment_method])[0]
    transaction_location_encoded = label_encoders['Transaction Location'].transform([transaction_location])[0]
    device_type_encoded = label_encoders['Device Type'].transform([device_type])[0]
    previous_fraud_encoded = label_encoders['Previous Fraudulent Activity'].transform([previous_fraud])[0]
    
    # Create a new user data point as a DataFrame with feature names
    user_data = pd.DataFrame([[transaction_amount, payment_method_encoded, transaction_location_encoded,
                               device_type_encoded, previous_fraud_encoded, time_between_transactions]],
                             columns=X.columns)  
    
    # Scale the numerical data using the pre-fitted scaler
    user_data_scaled = scaler.transform(user_data)
    
    return user_data_scaled

# Example of using the function
user_input = get_user_input()
print("Processed user input ready for prediction:", user_input)


# In[ ]:




