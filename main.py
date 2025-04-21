from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
import pandas as pd
from sklearn.linear_model import LinearRegression
from database import engine
from models import PredictionResponse
from modelImage import extract_feature
import requests

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sqlalchemy import text

from modelImage import extract_feature
from utils import load_features_from_folder, find_similar_images
import shutil
import os
from fastapi import UploadFile, File

app = FastAPI()



origins = [
    "http://localhost:5173",
    "http://localhost", 
    "https://coffeengonmoingay.shop"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

@app.get("/")
def root():
    return {"message": "Warehouse Prediction API đang chạy"}

def get_weather_data(month, year):
    api_url = f"https://api.weatherapi.com/v1/history.json?key=039e3c819a0340dc99a90412250504&q=Hanoi&dt={year}-{month:02d}-01"
    response = requests.get(api_url)
    data = response.json()
    avg_temperature = data['forecast']['forecastday'][0]['day']['avgtemp_c']
    return avg_temperature

# @app.get("/predict-restock", response_model=list[PredictionResponse])
# def predict_restock():
#     query = """
#     SELECT p.id as product_id, p.name as product_name, p.stock_quantity, c.name as category_name, p.image,
#            od.quantity, o.created_at, p.price
#     FROM order_details od
#     JOIN orders o ON od.orders_id = o.id
#     JOIN products p ON od.products_id = p.id
#     JOIN categories c ON c.id = p.categories_id
#     """
#     df = pd.read_sql(text(query), engine)

#     df['created_at'] = pd.to_datetime(df['created_at'])
#     df['month'] = df['created_at'].dt.to_period('M').astype(str)

#     df['is_holiday'] = df['month'].apply(lambda x: 1 if x in ['2025-01', '2025-12'] else 0)
#     df['is_promotion'] = df['month'].apply(lambda x: 1 if x in ['2025-03', '2025-06','2025-04','2025-05'] else 0) 

#     grouped = df.groupby(['product_id', 'product_name', 'category_name', 'image', 'month']).agg({
#         'quantity': 'sum', 'price': 'mean', 'is_holiday': 'mean', 'is_promotion': 'mean'
#     }).reset_index()

#     predictions = []

#     for product_id in grouped['product_id'].unique():
#         product_data = grouped[grouped['product_id'] == product_id]
#         # if len(product_data) < 2:
#         #     continue

#         product_data = product_data.copy()
#         product_data['month_index'] = range(len(product_data))

#         X = product_data[['month_index', 'price', 'is_holiday', 'is_promotion']]
#         y = product_data['quantity']
        
#         model = LinearRegression()
#         model.fit(X, y)

#         next_month_index = [[len(product_data), product_data['price'].iloc[-1], 
#                               product_data['is_holiday'].iloc[-1], product_data['is_promotion'].iloc[-1]]]
#         predicted_quantity = model.predict(next_month_index)[0]

#         current_stock = int(df[df['product_id'] == product_id]['stock_quantity'].iloc[0])
#         print(f"Predicted Quantity: {predicted_quantity}, Current Stock: {current_stock}")
#         predicted_needed = max(int(predicted_quantity), 0)

#         month = product_data['month'].iloc[-1].split('-')[1]
#         year = product_data['month'].iloc[-1].split('-')[0]
#         avg_temperature = get_weather_data(int(month), int(year))

#         temperature_adjustment_factor = 1 + (avg_temperature - 25) * 0.05
#         predicted_needed = int(predicted_needed * temperature_adjustment_factor)

#         predictions.append({
#             "id": int(product_id),
#             "name": product_data['product_name'].iloc[0],
#             "currentStock": current_stock,
#             "recommendedOrder": predicted_needed,
#             "image": product_data['image'].iloc[0],
#             "category": product_data['category_name'].iloc[0]
#         })

#     return predictions
@app.get("/predict-restock", response_model=list[PredictionResponse])
def predict_restock():
    query = """
    SELECT p.id as product_id, p.name as product_name, p.stock_quantity, c.name as category_name, p.image,
           od.quantity, o.created_at, p.price
    FROM order_details od
    JOIN orders o ON od.orders_id = o.id
    JOIN products p ON od.products_id = p.id
    JOIN categories c ON c.id = p.categories_id
    """
    df = pd.read_sql(text(query), engine)

    df['created_at'] = pd.to_datetime(df['created_at'])
    df['month'] = df['created_at'].dt.to_period('M').astype(str)

    df['is_holiday'] = df['month'].apply(lambda x: 1 if x in ['2025-01', '2025-12'] else 0)
    df['is_promotion'] = df['month'].apply(lambda x: 1 if x in ['2025-03', '2025-04', '2025-05', '2025-06'] else 0)

    grouped = df.groupby(['product_id', 'product_name', 'category_name', 'image', 'month']).agg({
        'quantity': 'sum',
        'price': 'mean',
        'is_holiday': 'mean',
        'is_promotion': 'mean'
    }).reset_index()

    predictions = []

    for product_id in grouped['product_id'].unique():
        product_data = grouped[grouped['product_id'] == product_id]

        product_data = product_data.copy()
        product_data['month_index'] = range(len(product_data))

        # if len(product_data) < 2:
        #     continue

        X = product_data[['month_index', 'price', 'is_holiday', 'is_promotion']]
        y = product_data['quantity']

        model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model.fit(X, y)

        next_month_index = [[len(product_data),
                             product_data['price'].iloc[-1],
                             product_data['is_holiday'].iloc[-1],
                             product_data['is_promotion'].iloc[-1]]]
        
        predicted_quantity = model.predict(next_month_index)[0]

        current_stock = int(df[df['product_id'] == product_id]['stock_quantity'].iloc[0])
        print(f"Predicted Quantity: {predicted_quantity}, Current Stock: {current_stock}")
        predicted_needed = max(int(predicted_quantity), 0)

        # Weather adjustment
        month = product_data['month'].iloc[-1].split('-')[1]
        year = product_data['month'].iloc[-1].split('-')[0]
        avg_temperature = get_weather_data(int(month), int(year))

        temperature_adjustment_factor = 1 + (avg_temperature - 25) * 0.05
        predicted_needed = int(predicted_needed * temperature_adjustment_factor)

        predictions.append({
            "id": int(product_id),
            "name": product_data['product_name'].iloc[0],
            "currentStock": current_stock,
            "recommendedOrder": predicted_needed,
            "image": product_data['image'].iloc[0],
            "category": product_data['category_name'].iloc[0]
        })

    return predictions

@app.post("/find_similar/")
async def find_similar_image(file: UploadFile = File(...), top_k: int = 5):
    IMAGE_FOLDER = "image_folder"
    all_features, image_paths = load_features_from_folder(IMAGE_FOLDER)
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        query_feature = extract_feature(temp_path)
        results = find_similar_images(query_feature, all_features, image_paths, top_k=top_k)
        return {"query_image": file.filename, "similar_images": results}
    finally:
        os.remove(temp_path)