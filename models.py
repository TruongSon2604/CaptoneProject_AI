from pydantic import BaseModel

class PredictionResponse(BaseModel):
    id: int
    name: str
    currentStock: int
    recommendedOrder: int
    image:str
    category:str