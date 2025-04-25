from sqlalchemy import create_engine
# from dotenv import load_dotenv
import os

# load_dotenv()

# DB_USER = os.getenv("DB_USER")
# DB_PASSWORD = os.getenv("DB_PASSWORD")
# DB_HOST = os.getenv("DB_HOST")
# DB_PORT = os.getenv("DB_PORT")
# DB_NAME = os.getenv("DB_NAME")

# DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

DB_USER = "qui34ckwg9fe_truongngocson2"
DB_PASSWORD = "Son26042003"
DB_HOST = "137.59.105.26"
DB_PORT = "3306"
DB_NAME = "qui34ckwg9fe_Doan2"

# Tạo URL kết nối
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
