import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 
import dagshub

dagshub.init(repo_owner='anshuyadav5056', repo_name='YT-MLOPS-Experiments--with-MLflow',mlflow=True)

token = "f99111413c6aa46b8c9c8b0de3156572982cb1cd"

mlflow.set_tracking_uri(
    f"https://anshuyadav5056:{token}@dagshub.com/"
    f"anshuyadav5056/YT-MLOPS-Experiments--with-MLflow.mlflow"
)

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target 

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 5
n_estimators = 8

#Mention your experiment below
mlflow.set_experiment('yT-MLops-Exp1')

with mlflow.start_run():
  rf = RandomForestClassifier(max_depth = max_depth,n_estimators=n_estimators, random_state=42)
  rf.fit(X_train, y_train)

  y_pred = rf.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)

  mlflow.log_metric('accuracy', accuracy)
  mlflow.log_param('max_depth', max_depth)
  mlflow.log_param('n_estimators', n_estimators)

  # Creating a confusion matrix plot
  cm = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(6,6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
  plt.ylabel('Actual')
  plt.xlabel('Predicted')
  plt.title('Confusion Matrix')

  # save plot
  plt.savefig("Confusion-matrix.png")

  # log artifacts using mlflow
  mlflow.log_artifact("Confusion-matrix.png")
  mlflow.log_artifact(__file__)

  #tags
  mlflow.set_tags({"Author": "Vikash", "Project": "Wine classification"})

  #log the model
  mlflow.sklearn.log_model(rf, "Random-forest-Model") 
  print(accuracy) 