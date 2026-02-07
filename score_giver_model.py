import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from pyproj import Transformer
from scipy.spatial import cKDTree
import joblib

# --- 1. LOAD DATA ---
print("Loading datasets...")
df_park = pd.read_csv('on-street-parking-bay-sensors.csv')
df_traffic = pd.read_csv('Traffic_Lights.csv')

# --- 2. CLEAN PARKING DATA ---
print("Cleaning Parking Data...")
# Drop rows with no Zone Number
df_park = df_park.dropna(subset=['zone_number'])

# Split Location into Lat/Lon
df_park[['Latitude', 'Longitude']] = df_park['location'].str.split(',', expand=True).astype(float)

# Convert Timestamp
df_park['status_timestamp'] = pd.to_datetime(df_park['status_timestamp'])
df_park['Hour'] = df_park['status_timestamp'].dt.hour
df_park['DayOfWeek'] = df_park['status_timestamp'].dt.dayofweek
df_park['Is_Weekend'] = df_park['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Create Target (1=Occupied, 0=Empty)
df_park['is_occupied'] = df_park['status_description'].apply(lambda x: 1 if x == 'Present' else 0)

# --- 3. CLEAN & CONVERT TRAFFIC DATA ---
print("Cleaning Traffic Data & Converting Coordinates...")
# The traffic data uses 'VicGrid94' (EPSG:3111). We convert to 'WGS84' (EPSG:4326) aka Lat/Lon.
transformer = Transformer.from_crs("epsg:3111", "epsg:4326")

# Apply conversion
# Note: VicGrid is usually (East, North), Output is (Lat, Lon)
traffic_lat, traffic_lon = transformer.transform(df_traffic['X'].values, df_traffic['Y'].values)
df_traffic['Latitude'] = traffic_lat
df_traffic['Longitude'] = traffic_lon

# Keep only valid coordinates
df_traffic = df_traffic.dropna(subset=['Latitude', 'Longitude'])

# --- 4. MERGE DATA (Spatial Join) ---
print("Calculating Traffic Scores (Distance to nearest traffic light)...")

# We use cKDTree to find the nearest traffic light for every parking spot efficiently
traffic_coords = df_traffic[['Latitude', 'Longitude']].values
parking_coords = df_park[['Latitude', 'Longitude']].values

tree = cKDTree(traffic_coords)
distances, indices = tree.query(parking_coords, k=1) # k=1 means find the 1 nearest neighbor

# Add distance to parking dataframe (Distance is in degrees, approx conversion to meters is * 111,000)
df_park['Dist_to_Light_deg'] = distances

# --- 5. CALCULATE TRAFFIC SCORE ---
# Hypothesis: Closer to light = More Traffic = "Bad" for parking ease, but "Good" for finding a spot?
# Let's say: Traffic Score 100 = Very Busy (Close to light), 0 = Quiet (Far from light)
scaler = MinMaxScaler()
# Invert distance so closer = higher score
df_park['Traffic_Score'] = 1 - scaler.fit_transform(df_park[['Dist_to_Light_deg']])

print("Data merged successfully!")
print(df_park[['zone_number', 'Dist_to_Light_deg', 'Traffic_Score']].head())

# --- 6. TRAIN AVAILABILITY MODEL ---
print("Training Availability Model...")
features = ['Hour', 'DayOfWeek', 'Is_Weekend', 'Latitude', 'Longitude', 'Traffic_Score']
X = df_park[features]
y = df_park['is_occupied']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# --- 7. SAVE EVERYTHING ---
joblib.dump(model, 'complex_parking_model.pkl')
# Save the traffic scaler to normalize future data
joblib.dump(scaler, 'traffic_scaler.pkl') 
# Save the traffic tree to find distances for new spots
joblib.dump(tree, 'traffic_tree.pkl') 

print("System Ready. All files saved.")