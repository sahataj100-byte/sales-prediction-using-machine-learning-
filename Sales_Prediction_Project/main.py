import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv("Advertising.csv")

print("=" * 50)
print("        ADVERTISING DATASET")
print("=" * 50)

# Remove unwanted column
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# ===============================
# DATA CLEANING
# ===============================

print("\nFirst 5 Rows\n")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Values:", df.duplicated().sum())

print("\nDataset Information\n")
print(df.info())

print("\nStatistical Summary\n")
print(df.describe())

# ===============================
# FEATURE SELECTION
# ===============================

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# ===============================
# TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# MODEL TRAINING
# ===============================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# ===============================
# PREDICTION
# ===============================

y_pred = model.predict(X_test)

# ===============================
# MODEL EVALUATION
# ===============================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("MAE :", round(mae,2))
print("MSE :", round(mse,2))
print("RMSE:", round(rmse,2))
print("R2 Score:", round(r2,2))

# ===============================
# FUTURE SALES PREDICTION
# ===============================

tv = float(input("\nEnter TV Advertising Budget : "))
radio = float(input("Enter Radio Advertising Budget : "))
newspaper = float(input("Enter Newspaper Advertising Budget : "))

future = model.predict([[tv, radio, newspaper]])

print("\nPredicted Sales =", round(future[0],2))

# ===============================
# GRAPH 1
# TV vs SALES
# ===============================

plt.figure(figsize=(8,5))
plt.scatter(df["TV"], df["Sales"])
plt.title("TV Advertising vs Sales")
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# ===============================
# GRAPH 2
# RADIO vs SALES
# ===============================

plt.figure(figsize=(8,5))
plt.scatter(df["Radio"], df["Sales"])
plt.title("Radio Advertising vs Sales")
plt.xlabel("Radio Advertising")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# ===============================
# GRAPH 3
# NEWSPAPER vs SALES
# ===============================

plt.figure(figsize=(8,5))
plt.scatter(df["Newspaper"], df["Sales"])
plt.title("Newspaper Advertising vs Sales")
plt.xlabel("Newspaper Advertising")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# ===============================
# GRAPH 4
# SALES DISTRIBUTION
# ===============================

plt.figure(figsize=(8,5))
plt.hist(df["Sales"], bins=20)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# ===============================
# GRAPH 5
# ACTUAL vs PREDICTED
# ===============================

plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Sales")
plt.plot(y_pred, label="Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Test Data")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# GRAPH 6
# FEATURE IMPORTANCE
# ===============================

importance = pd.Series(model.coef_, index=X.columns)

plt.figure(figsize=(8,5))
importance.plot(kind="bar")
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Coefficient")
plt.grid(True)
plt.show()

# ===============================
# GRAPH 7
# BOXPLOT
# ===============================

plt.figure(figsize=(8,5))
df.boxplot(column=["TV","Radio","Newspaper","Sales"])
plt.title("Box Plot")
plt.show()

# ===============================
# GRAPH 8
# CORRELATION HEATMAP
# ===============================

corr = df.corr()

plt.figure(figsize=(8,6))
plt.imshow(corr, cmap="coolwarm")

plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.colorbar()

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, round(corr.iloc[i, j],2),
                 ha="center", va="center", color="black")

plt.title("Correlation Heatmap")
plt.show()