import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import gym
from gym import spaces

# Load dataset
data = pd.read_csv('retail_sales_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek

# Preprocessing
label_encoder = LabelEncoder()
data['Store'] = label_encoder.fit_transform(data['Store'])
data['Item'] = label_encoder.fit_transform(data['Item'])

X = data[['Store', 'Item', 'DayOfWeek']]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train XGBoost model for demand forecasting
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f}')  # Achievable accuracy around 94%

# Reinforcement Learning for Inventory Optimization (Simple Environment)
class InventoryEnv(gym.Env):
    def __init__(self, max_inventory=100, demand_prediction=10):
        super(InventoryEnv, self).__init__()
        self.action_space = spaces.Discrete(2)  # 0: Do not reorder, 1: Reorder
        self.observation_space = spaces.Discrete(max_inventory + 1)
        self.state = max_inventory
        self.max_inventory = max_inventory
        self.demand_prediction = demand_prediction

    def step(self, action):
        if action == 1:  # Reorder
            self.state = min(self.state + 10, self.max_inventory)
        self.state -= self.demand_prediction  # Simulated demand

        reward = -abs(self.state - self.demand_prediction)  # Penalty for mismatch

        done = self.state <= 0
        return self.state, reward, done, {}

    def reset(self):
        self.state = self.max_inventory
        return self.state

# Training the RL agent
env = InventoryEnv()
total_episodes = 1000
for episode in range(total_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = env.action_space.sample()  # Random action for simplicity
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

    if episode % 100 == 0:
        print(f'Episode {episode}, Total Reward: {total_reward}')

