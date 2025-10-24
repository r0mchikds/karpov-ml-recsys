from typing import List
from pydantic import BaseModel


# Модель поста в ответе API
class PostGet(BaseModel):
    id: int # ID поста
    text: str # Текст поста
    topic: str # Тематика поста

    class Config:
        orm_mode = True # Позволяет использовать ORM-объекты (например, из SQLAlchemy)

# Модель основного ответа API
class Response(BaseModel):
    exp_group: str # Группа эксперимента ("control" или "test")
    recommendations: List[PostGet] # Список рекомендованных постов
