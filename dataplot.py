import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("liveness_features.csv")

plt.hist(df[df.label=="REAL"]["score"], bins=30, alpha=0.6, label="REAL")
plt.hist(df[df.label=="SPOOF"]["score"], bins=30, alpha=0.6, label="SPOOF")
plt.legend()
plt.title("Final Liveness Score Distribution")
plt.show()

from sklearn.metrics import roc_curve

y = (df.label == "REAL").astype(int)
scores = df.score

fpr, tpr, thresholds = roc_curve(y, scores)

# choose threshold at FAR ≈ 1%
target_far = 0.01
idx = (fpr <= target_far).nonzero()[0][-1]
tau_live = thresholds[idx]
print("Calibrated tau_live:", tau_live)

