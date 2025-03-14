import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the dataset
file_path = r"C:\Users\teddy\Documents\Grad School\ECGR 4127_5127 - Machine Learning for the Internet of Things\Homeworks\HW4\hw4_data.csv"
df = pd.read_csv(file_path)

# Extract necessary columns
model_output = df["model_output"].to_numpy()
true_class = df["true_class"].to_numpy()
y_pred = df["prediction"].to_numpy()

# Compute Confusion Matrix Values
TP = ((y_pred == 1) & (true_class == 1)).sum()
FP = ((y_pred == 1) & (true_class == 0)).sum()
TN = ((y_pred == 0) & (true_class == 0)).sum()
FN = ((y_pred == 0) & (true_class == 1)).sum()

# Compute Precision and Recall
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

# Display results
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

# Compute and Plot ROC Curve
fpr, tpr, thresholds = roc_curve(true_class, model_output)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# Find the Minimum FPR to Achieve at Least 90% TPR
desired_tpr = 0.90
valid_indices = np.where(tpr >= desired_tpr)[0]

if len(valid_indices) > 0:
    min_fpr = np.min(fpr[valid_indices])
    print(f"Minimum False Positive Rate to achieve at least 90% TPR: {min_fpr:.3f}")
else:
    print("Could not achieve 90% TPR with any threshold.")
