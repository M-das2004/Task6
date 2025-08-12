import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- Step 1: Load the dataset and normalize features ---
# The file "Iris.csv" is accessible in the environment.
try:
    df = pd.read_csv("Iris.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Iris.csv not found.")
    exit()

# Drop the 'Id' column as it's not useful for classification.
df = df.drop('Id', axis=1)

# Separate features (X) and target (y)
X = df.drop('Species', axis=1)
y = df['Species']

# Normalize the features. This is a crucial step for distance-based algorithms like KNN.
# StandardScaler will scale the data to have a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeatures have been normalized.")
print("Original features shape:", X.shape)
print("Normalized features shape:", X_scaled.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# --- Step 2 & 3: Use KNeighborsClassifier and experiment with different K values ---
# We'll test a range of K values to see which one gives the best accuracy.
k_values = range(1, 21)
accuracy_scores = []

for k in k_values:
    # Create and train the KNN model with the current k
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = knn.predict(X_test)
    
    # Calculate the accuracy and store it
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    print(f"Accuracy for K={k}: {accuracy:.4f}")

# Find the best K value
best_k_index = np.argmax(accuracy_scores)
best_k = k_values[best_k_index]
print(f"\nBest K value found: {best_k} with an accuracy of {accuracy_scores[best_k_index]:.4f}")

# --- Visualize the results of different K values ---
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='--')
plt.title('Accuracy vs. K Value')
plt.xlabel('K Value (Number of Neighbors)')
plt.ylabel('Accuracy Score')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# --- Step 4: Evaluate the model with the best K ---
# Now, let's train a final model using the best K we found.
final_knn_model = KNeighborsClassifier(n_neighbors=best_k)
final_knn_model.fit(X_train, y_train)

# Get predictions from the final model
y_pred_final = final_knn_model.predict(X_test)

# Print a classification report for detailed metrics (precision, recall, f1-score)
print("\n--- Final Model Evaluation ---")
print(f"Classification Report for K={best_k}:\n")
print(classification_report(y_test, y_pred_final))

# Generate and print the confusion matrix
cm = confusion_matrix(y_test, y_pred_final)
print(f"\nConfusion Matrix for K={best_k}:\n")
print(cm)

# --- Step 5: Visualize the confusion matrix ---
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title(f'Confusion Matrix for K={best_k}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

