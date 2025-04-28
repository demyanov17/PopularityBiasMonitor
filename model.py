import os
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares

class RecommendationGenerator:

    def __init__(self, data_dir, factors=50, regularization=0.01, iterations=15, top_n=10):
        self.data_dir = data_dir
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.top_n = top_n

    def generate_recommendations(self):
        # Iterate through all folders with date batches
        for folder in sorted(os.listdir(self.data_dir)):
            folder_path = os.path.join(self.data_dir, folder)
            ratings_path = os.path.join(folder_path, 'training_data.csv')
            print(ratings_path)
            print(os.path.exists(ratings_path))

            if os.path.isdir(folder_path) and os.path.exists(ratings_path):
                print(f"Обработка папки: {folder}")

                ratings = pd.read_csv(ratings_path)

                # Build user-item sparse matrix
                user_mapping = {user_id: idx for idx, user_id in enumerate(ratings['user_id'].unique())}
                item_mapping = {item_id: idx for idx, item_id in enumerate(ratings['item_id'].unique())}
                reverse_user_mapping = {idx: user_id for user_id, idx in user_mapping.items()}
                reverse_item_mapping = {idx: item_id for item_id, idx in item_mapping.items()}

                # Map user_id and item_id to internal indices
                user_idx = ratings['user_id'].map(user_mapping)
                item_idx = ratings['item_id'].map(item_mapping)

                # Ensure all users and items are correctly mapped
                if user_idx.isnull().any() or item_idx.isnull().any():
                    print(f"Ошибка: не все user_id или item_id маппируются корректно.")
                    continue  # Пропускаем данный файл, если есть ошибки маппинга

                # Build the interaction sparse matrix
                interactions = sparse.coo_matrix(
                    (ratings['interaction'], (user_idx, item_idx)),
                    shape=(len(user_mapping), len(item_mapping))
                ).tocsr()

                # Train ALS model
                model = AlternatingLeastSquares(
                    factors=self.factors,
                    regularization=self.regularization,
                    iterations=self.iterations,
                    use_gpu=False
                )
                model.fit(interactions)

                ratings_path = os.path.join(folder_path, 'ratings.csv')

                # Build user-item sparse matrix again for final ratings
                user_mapping = {user_id: idx for idx, user_id in enumerate(ratings['user_id'].unique())}
                item_mapping = {item_id: idx for idx, item_id in enumerate(ratings['item_id'].unique())}
                reverse_user_mapping = {idx: user_id for user_id, idx in user_mapping.items()}
                reverse_item_mapping = {idx: item_id for item_id, idx in item_mapping.items()}

                # Map user_id and item_id to internal indices
                user_idx = ratings['user_id'].map(user_mapping)
                item_idx = ratings['item_id'].map(item_mapping)

                # Ensure all users and items are correctly mapped
                if user_idx.isnull().any() or item_idx.isnull().any():
                    print(f"Ошибка: не все user_id или item_id маппируются корректно.")
                    continue # Skip this file if there are mapping issues

                # Build the interaction sparse matrix
                interactions = sparse.coo_matrix(
                    (ratings['interaction'], (user_idx, item_idx)),
                    shape=(len(user_mapping), len(item_mapping))
                ).tocsr()

                # Generate recommendations
                recommendations = []

                # For each user, filter the interaction matrix
                for user_internal_id in range(len(user_mapping)):
                    user_interactions = interactions[user_internal_id, :].toarray().flatten()
                    
                    # Check if user has any interactions
                    if np.sum(user_interactions) > 0:
                        # Generate recommendations for the user
                        recommended = model.recommend(user_internal_id, interactions[user_internal_id], N=self.top_n, filter_already_liked_items=True)
                        
                        # Recommendations are returned as (item_internal_id, score) tuples
                        for recommendation in recommended:
                            item_internal_id = int(recommendation[0])  # Преобразуем в целое число
                            score = recommendation[1]
                            recommendations.append({
                                'user_id': reverse_user_mapping[user_internal_id],
                                'item_id': reverse_item_mapping[item_internal_id],
                                'score': score
                            })
                    else:
                        print(f"Пользователь {reverse_user_mapping[user_internal_id]} не имеет взаимодействий.")

                # Save recommendations to a file if any were created
                if recommendations:
                    recs_df = pd.DataFrame(recommendations)
                    recs_df.to_csv(os.path.join(folder_path, 'recommendations.csv'), index=False)
                    print(f"Сохранено {len(recommendations)} рекомендаций в {folder_path}")
                else:
                    print(f"Нет рекомендаций для пользователей в папке {folder}")

if __name__ == "__main__":
    generator = RecommendationGenerator(data_dir='batches')
    generator.generate_recommendations()
