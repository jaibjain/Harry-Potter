import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib

# -------------------------
# Load dataset
# -------------------------

X = []
y = []

files = glob.glob("*.npy")

if len(files) == 0:
    print("No .npy files found!")
    exit()

for file in files:
    data = np.load(file)
    
    # Spell name is before first underscore
    label = os.path.basename(file).split("_")[0]
    
    X.append(data)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset size:", len(X))
print("Spells:", np.unique(y))

# -------------------------
# Train / Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Train Model
# -------------------------

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------

y_pred = clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------
# Save Model
# -------------------------

joblib.dump(clf, "spell_classifier.joblib")
print("\nModel saved as spell_classifier.joblib")