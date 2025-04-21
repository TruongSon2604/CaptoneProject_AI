from sqlalchemy import create_engine

# DB_USER = "root"
# DB_PASSWORD = ""
# DB_HOST = "127.0.0.1"
# DB_PORT = "3307"
# DB_NAME = "captoneproject"

DB_USER = "qui34ckwg9fe_truongngocson2"
DB_PASSWORD = "Son26042003"
DB_HOST = "137.59.105.26"
DB_PORT = "3306"
DB_NAME = "qui34ckwg9fe_Doan2"

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
