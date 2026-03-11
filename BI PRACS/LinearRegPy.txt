import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# [1] Input Data using a DataFrame
# We create a dictionary and convert it straight into a DataFrame
data = {
    'Height': [151, 174, 138, 186, 128, 136, 179, 163, 152, 131], #
    'Weight': [63, 81, 56, 91, 47, 57, 76, 72, 62, 48]            #
}
df = pd.DataFrame(data)

# [2] Prepare X (Predictor) and y (Response)
# Pro-tip: Using double brackets [['Height']] keeps X as a 2D DataFrame, which scikit-learn requires!
X = df[['Height']] 
y = df['Weight']

# [3] Create and Train the Model
model = LinearRegression()
model.fit(X, y)

print(f"Coefficient (a - slope): {model.coef_[0]:.4f}")
print(f"Intercept (b - constant): {model.intercept_:.4f}")

# [4] Predict the weight of a new person
# We create a tiny new DataFrame for the 170cm person
new_person = pd.DataFrame({'Height': [170]}) 
predicted_weight = model.predict(new_person)

print(f"\nPredicted weight for 170cm height: {predicted_weight[0]:.2f} Kg")

# [5] Visualize
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Best Fit Line')
plt.title("Height & Weight Regression")
plt.xlabel("Height in cm")
plt.ylabel("Weight in Kg")
plt.legend()
plt.show()