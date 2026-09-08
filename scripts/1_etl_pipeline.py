import pandas as pd

def main():
    print("1. Loading datasets from Hugging Face...")

    # Читаем тексты
    df_texts = pd.read_parquet('../data/raw/datasets1.parquet', columns=['isbn', 'title', 'desc', 'author'])
    # Читаем метаданные (таблица-мост)
    df_meta = pd.read_csv('../data/raw/datasets2.csv', usecols=['isbn', 'work_id', 'genres', 'image_url'])
    # Читаем отзывы
    df_reviews = pd.read_csv('../data/raw/reviews.csv', usecols=['user_id', 'work_id', 'rating'])

    print("2. Cleaning keys before merging...")
    # Удаляем строки без идентификаторов, чтобы избежать ошибок при слиянии
    df_texts = df_texts.dropna(subset=['isbn'])
    df_meta = df_meta.dropna(subset=['isbn', 'work_id'])
    df_reviews = df_reviews.dropna(subset=['user_id', 'work_id', 'rating'])

    print("3. Building the unified graph (INNER JOIN)...")
    # Соединяем тексты и метаданные по ISBN
    df_books = pd.merge(df_texts, df_meta, on='isbn', how='inner')
    # Соединяем результат предыдущего шага с оценками по id
    df_full = pd.merge(df_reviews, df_books, on='work_id', how='inner')
    print(f"Dataset size before filtering: {df_full.shape[0]} reviews.")

    print("4. Eliminating 'cold start' (filtering low-activity records)...")
    # Оставляем только популярные книги
    book_counts = df_full['work_id'].value_counts()
    popular_books = book_counts[book_counts >= 10].index
    df_filtered = df_full[df_full['work_id'].isin(popular_books)]
    # Оставляем только активных пользователей
    user_counts = df_filtered['user_id'].value_counts()
    active_users = user_counts[user_counts >= 5].index
    df_filtered = df_filtered[df_filtered['user_id'].isin(active_users)]
    print(f"Dataset size after filtering: {df_filtered.shape[0]} reviews.")

    print("5. Formatting final tables for PostgreSQL database...")
    # Таблица Books без дубликатов
    final_books = df_filtered[['work_id', 'title', 'author', 'desc', 'genres', 'image_url']].drop_duplicates(subset=['work_id'])
    # Переименовываем колонки для SQL-скрипта
    final_books = final_books.rename(columns={'work_id': 'goodreads_book_id', 'desc': 'description'})
    # Таблица Interactions
    final_interactions = df_filtered[['user_id', 'work_id', 'rating']]
    final_interactions = final_interactions.rename(columns={'work_id': 'goodreads_book_id'})
    print(f"Total ready for DB loading: Books - {final_books.shape[0]}, Reviews - {final_interactions.shape[0]}")

    # Сохраняем результат
    final_books.to_csv('../data/processed/books_final.csv', index=False)
    final_interactions.to_csv('../data/processed/interactions_final.csv', index=False)
    print("Success! Files books_final.csv and interactions_final.csv have been saved in the data/processed/ folder.")

if __name__ == "__main__":
    main()