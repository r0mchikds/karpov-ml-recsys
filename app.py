import pandas as pd
import numpy as np
from typing import List
from schema import PostGet, Response
from sqlalchemy import create_engine
from catboost import CatBoostClassifier
import os
from fastapi import FastAPI
from datetime import datetime
import hashlib
# from loguru import logger # (опционально для логирования)


# Константы для A/B-разделения
EXPERIMENT_SALT = "rSecr3t" # Соль для хеширования
TEST_GROUP_PERCENTAGE = 50 # Процент пользователей в тестовой группе

app = FastAPI()

# ЗАГРУЗКА МОДЕЛЕЙ

# Определяем путь к модели (локально или в LMS)
def get_model_path(name: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        return f"/workdir/user_input/{name}" # Путь для среды LMS
    else:
        return name # Локальный путь

# Загружаем обе модели (control / test)
def load_models():
    model_control = CatBoostClassifier()
    model_control.load_model(get_model_path("model_control"))
    
    model_test = CatBoostClassifier()
    model_test.load_model(get_model_path("model_test"))
    
    return model_control, model_test

# ЗАГРУЗКА ПРИЗНАКОВ

# Универсальная функция постраничной загрузки SQL-таблицы
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

# Загружаем все таблицы фичей из базы
def load_features() -> pd.DataFrame:
    # Отберем те комбинации записей user_id и post_id,
    # где уже стоит лайк,
    # чтобы не рекомендовать повторно понравившиеся посты
    query_posts_liked = """
    SELECT DISTINCT user_id, post_id
    FROM public.feed_data
    WHERE action = 'like'
    """
    posts_liked = batch_load_sql(query_posts_liked)
    
    # Фичи пользователей
    query_user = "SELECT * FROM tikhonovrs96_features_user_lesson_22"
    user_features = pd.read_sql(
        query_user,
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
            "postgres.lab.karpov.courses:6432/startml"
    )

    # Фичи постов TF-IDF
    query_post_tfidf = "SELECT * FROM tikhonovrs96_features_post_tfidf_lesson_22"
    post_features_tfidf = pd.read_sql(
        query_post_tfidf,
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
            "postgres.lab.karpov.courses:6432/startml"
    )

    # Фичи постов BERT
    query_post_bert = "SELECT * FROM tikhonovrs96_features_post_bert_lesson_22"
    post_features_bert = pd.read_sql(
        query_post_bert,
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
            "postgres.lab.karpov.courses:6432/startml"
    )

    # Mean Target Encoding
    query_mte = "SELECT * FROM tikhonovrs96_features_mte_lesson_22"
    mte = pd.read_sql(
        query_mte,
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
            "postgres.lab.karpov.courses:6432/startml"
    )

    return posts_liked, user_features, post_features_tfidf, post_features_bert, mte

# ИНИЦИАЛИЗАЦИЯ

# Загружаем модели
model_control, model_test = load_models()

# Определяем экспериментальную группу по user_id (через хеш)
def get_exp_group(user_id: int) -> str:
    value = f"{user_id}_{EXPERIMENT_SALT}"
    hash_digest = hashlib.md5(value.encode()).hexdigest()
    group_num = int(hash_digest, 16) % 100
    return "test" if group_num < TEST_GROUP_PERCENTAGE else "control"

# Загружаем все фичи
features = load_features()

# Подготавливаем вспомогательные словари для быстрого доступа
liked_posts_dict = features[0].groupby('user_id')['post_id'].apply(set).to_dict()

mte_dict = (
    features[4].groupby("feature_name")
    .apply(lambda df: df.set_index("feature_value")["encoded_value"].to_dict())
    .to_dict()
)

post_features_dict = {
    "control": features[2],
    "test": features[3]
}

# ДОБАВЛЕНИЕ ФИЧЕЙ ДАТ И ВРЕМЕНИ

# Добавляем временные признаки (dayofweek, sin/cos день/месяц)
def add_date_features(df: pd.DataFrame, date_obj: datetime):
    # OHE для дня недели
    dow = date_obj.weekday()
    for i in range(1, 7):
        df[f"dayofweek_{i}"] = 1 if dow == i else 0
    
    # Циклическое кодирование для дня
    day = date_obj.day
    df['day_sin'] = np.sin(2 * np.pi * day / 31)
    df['day_cos'] = np.cos(2 * np.pi * day / 31)

    # Циклическое кодирование для месяца
    month = date_obj.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

    return df

# РЕКОМЕНДАЦИИ

# Основная функция построения рекомендаций
def recommend(user_id: int, time: datetime, limit: int,
              model: CatBoostClassifier, post_features: pd.DataFrame, group: str) -> Response:
    content = post_features[['post_id', 'text', 'topic']]
    post_features = post_features.drop(['text', 'topic'], axis=1)

    # Получаем признаки пользователя
    user_features = features[1][features[1]['user_id'] == user_id].drop('user_id', axis=1)
    
    # Дублируем фичи пользователя для всех постов
    user_array = np.tile(user_features.values, (post_features.shape[0], 1))
    user_expanded = pd.DataFrame(user_array, columns=user_features.columns)
    user_post_features = pd.concat((user_expanded, post_features), axis=1).set_index('post_id')

    # Добавляем MTE-кодировки
    for col in ['city', 'country', 'age']:
        user_post_features[f"{col}_mte"] = user_post_features[col].map(mte_dict[col])
        default_val = mte_dict[col].get("__DEFAULT__", 0.0)
        user_post_features[f"{col}_mte"] = user_post_features[f"{col}_mte"].fillna(default_val)
        user_post_features = user_post_features.drop(col, axis=1)

    # Добавляем календарные фичи
    user_post_features = add_date_features(user_post_features, time)
    
    # Предсказания вероятностей лайка
    preds = model.predict_proba(user_post_features)[:, 1]
    user_post_features["preds"] = preds

    # Убираем уже лайкнутые посты
    posts_liked = liked_posts_dict.get(user_id, set())
    filtered_posts = user_post_features[~(user_post_features.index.isin(posts_liked))]
    
    # Топ-N по предсказаниям
    recommended_posts = filtered_posts.sort_values('preds', ascending=False).head(limit).index
    
    # Сопоставляем с текстами постов
    content_dict = content.set_index("post_id").to_dict(orient="index")

    # Формируем ответ API
    return Response(
        exp_group=group,
        recommendations=[
            PostGet(id=i,
                    text=content_dict[i]["text"],
                    topic=content_dict[i]["topic"]) for i in recommended_posts
        ]
    )


# Вспомогательные функции для каждой группы
# Контрольной
def recommend_control(user_id: int, time: datetime, limit: int) -> Response:
    model = model_control
    post_features = post_features_dict["control"]
    return recommend(user_id, time, limit, model, post_features, "control")

# Тестовой
def recommend_test(user_id: int, time: datetime, limit: int) -> Response:
    model = model_test
    post_features = post_features_dict["test"]
    return recommend(user_id, time, limit, model, post_features, "test")

# Определяем, какую группу использовать для текущего пользователя
def get_recommendations(user_id: int, time: datetime, limit: int) -> Response:
    group = get_exp_group(user_id)
    if group == "control":
        return recommend_control(user_id, time, limit)
    elif group == "test":
        return recommend_test(user_id, time, limit)
    else:
        raise ValueError("unknown group")

# Эндпоинт выдачи рекомендаций пользователю
@app.get("/post/recommendations/", response_model=Response)
def give_recommendations(id: int, time: datetime, limit: int = 5) -> Response:
    return get_recommendations(id, time, limit)
