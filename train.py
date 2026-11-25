import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from features import engineer, make_preprocessor
df = pd.read_csv("data/train.csv")
X = engineer(df)
y = df["Transported"].astype(int)
model = Pipeline([
    ("prep", make_preprocessor()),
    ("clf", LGBMClassifier(random_state=42, verbose=-1)),
])
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
model.fit(X, y)
joblib.dump(model, "model.joblib")
print("Saved model.joblib")