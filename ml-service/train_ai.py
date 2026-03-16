import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv('transactions.csv')
X = df.drop('fraud', axis=1)
y = df['fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and save the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
joblib.dump(model, 'fraud_model.pkl')
print("✅ AI Model ready: fraud_model.pkl")