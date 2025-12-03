from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from dotenv import load_dotenv
import os

load_dotenv(".env.local")
DATABASE_URL = os.getenv("DATABASE_URL")

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    label = Column(Integer, nullable=False)
    label_name = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    probs = Column(String)  
    advice = Column(String)  
    image_data = Column(Text)  
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)