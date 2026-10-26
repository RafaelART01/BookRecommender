import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Отключаем лишние предупреждения для чистоты консоли
warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


def load_and_preprocess_data():
    print("1. Loading cleaned datasets...")
    books = pd.read_csv('../data/processed/books_final_clean.csv')
    interactions = pd.read_csv('../data/processed/interactions_final.csv')
    books['description'] = books['description'].fillna('')
    positive_interactions = interactions[interactions['rating'] >= 3].copy()
    return books, positive_interactions


def create_train_test_split(interactions, test_ratio=0.2):
    print("2. Splitting user interactions into Train and Test sets...")
    train_list, test_list = [], []
    for user_id, group in interactions.groupby('user_id'):
        n_items = len(group)
        if n_items >= 5:
            n_test = max(1, int(n_items * test_ratio))
            train_list.append(group.iloc[:-n_test])
            test_list.append(group.iloc[-n_test:])

    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    print(f"   Train interactions: {len(train_df)}, Test interactions: {len(test_df)}")
    return train_df, test_df


def compute_similarity_matrix_tfidf(books):
    print("3. Computing similarities via TF-IDF (Content-based Baseline)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(books['description'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


def compute_similarity_matrix_transformer(books, model_name='all-mpnet-base-v2'):
    print(f"4. Computing similarities via Heavy Transformer ({model_name})...")
    # Эта модель скачает ~420 МБ весов, может занять пару минут
    model = SentenceTransformer(model_name)
    embeddings = model.encode(books['description'].tolist(), batch_size=32, show_progress_bar=True,
                              convert_to_numpy=True)
    return cosine_similarity(embeddings, embeddings)


def compute_similarity_matrix_collaborative(books, train_df):
    print("5. Computing similarities via Collaborative Filtering (User-based)...")

    # Используем pivot_table вместо pivot, чтобы усреднить дублирующиеся оценки
    user_item_matrix = train_df.pivot_table(
        index='goodreads_book_id',
        columns='user_id',
        values='rating',
        aggfunc='mean'
    ).fillna(0)

    # Синхронизируем порядок строк с нашим датафреймом books
    full_books_df = pd.DataFrame(index=books['goodreads_book_id'])
    full_matrix = full_books_df.join(user_item_matrix).fillna(0)

    # Считаем косинусное сходство между книгами на основе оценок пользователей
    return cosine_similarity(full_matrix.values, full_matrix.values)


def evaluate_recommender(books, train_df, test_df, sim_matrix, k=10):
    book_id_to_idx = {bid: idx for idx, bid in enumerate(books['goodreads_book_id'])}
    precisions = []
    test_grouped = test_df.groupby('user_id')['goodreads_book_id'].apply(set).to_dict()

    for user_id, train_group in train_df.groupby('user_id'):
        if user_id not in test_grouped:
            continue
        test_books = test_grouped[user_id]
        train_indices = [book_id_to_idx[bid] for bid in train_group['goodreads_book_id'].tolist() if
                         bid in book_id_to_idx]
        if not train_indices:
            continue

        user_scores = sim_matrix[train_indices].mean(axis=0)
        for idx in train_indices:
            user_scores[idx] = -1.0

        top_k_indices = np.argsort(user_scores)[::-1][:k]
        recommended_books = set(books.iloc[top_k_indices]['goodreads_book_id'])
        hits = len(recommended_books & test_books)
        precisions.append(hits / k)

    return np.mean(precisions) if precisions else 0.0


def save_comparison_plot(results_dict):
    print("6. Generating evaluation plot...")
    models = list(results_dict.keys())
    scores = [results_dict[m] for m in models]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(models, scores, color=['#95a5a6', '#3498db', '#2ecc71'], width=0.5)

    plt.ylabel('Precision@10', fontsize=12)
    plt.title('Comparison of Recommendation Models', fontsize=14)
    plt.ylim(0, max(scores) * 1.3 if max(scores) > 0 else 1.0)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.001, f"{yval:.4f}", ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = '../data/processed/metrics_result.png'
    plt.savefig(plot_path, dpi=300)
    print(f"   Plot successfully saved to: {plot_path}")


def main():
    books, interactions = load_and_preprocess_data()
    train_df, test_df = create_train_test_split(interactions, test_ratio=0.2)

    # 1. TF-IDF
    tfidf_sim = compute_similarity_matrix_tfidf(books)
    tfidf_prec = evaluate_recommender(books, train_df, test_df, tfidf_sim, k=10)
    print(f"\n[Result] TF-IDF Precision@10: {tfidf_prec:.4f}")

    # 2. Heavy Transformer (MPNet)
    mpnet_sim = compute_similarity_matrix_transformer(books, model_name='all-mpnet-base-v2')
    mpnet_prec = evaluate_recommender(books, train_df, test_df, mpnet_sim, k=10)
    print(f"[Result] MPNet Precision@10: {mpnet_prec:.4f}")

    # 3. Collaborative Filtering (User Ratings)
    cf_sim = compute_similarity_matrix_collaborative(books, train_df)
    cf_prec = evaluate_recommender(books, train_df, test_df, cf_sim, k=10)
    print(f"[Result] Collaborative Filtering Precision@10: {cf_prec:.4f}\n")

    save_comparison_plot({
        'TF-IDF\n(Text words)': tfidf_prec,
        'Transformer MPNet\n(Text semantics)': mpnet_prec,
        'Collaborative\n(User Ratings)': cf_prec
    })
    print("Experiment completed successfully!")


if __name__ == "__main__":
    main()