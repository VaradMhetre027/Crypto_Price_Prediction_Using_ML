# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Load the dataset
# Replace 'bitcoin_historical_data.csv' with the path to your dataset
df = pd.read_csv("C:/Users/DELL/Desktop/my project/bitcoin.csv")

# Step 2: Preprocess the data
# Convert Date to datetime and sort the data
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Feature engineering
df['Price_Change'] = df['Close'].pct_change() * 100  # Percentage change
df['Rolling_Mean'] = df['Close'].rolling(window=7).mean()  # 7-day moving average
df['Rolling_Std'] = df['Close'].rolling(window=7).std()  # 7-day standard deviation

# Drop rows with missing values
df = df.dropna()

# Define features and target
X = df[['Open', 'High', 'Low', 'Volume', 'Price_Change', 'Rolling_Mean', 'Rolling_Std']]
y = df['Close']

# Step 3: Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# Step 8: Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(y_pred, label='Predicted Prices', color='red')
plt.legend()
plt.title("Bitcoin Price Prediction")
plt.xlabel("Data Points")
plt.ylabel("Price")
plt.show()
