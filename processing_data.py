import os
import pandas as pd
from datetime import timedelta


class DataProcessor:

    def __init__(self, min_user_interactions=5, min_item_interactions=10):
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Universal function for loading a ratings file from MovieLens.
        Supports both .dat and .csv formats.

        Args:
            file_path (str): Path to the file.

        Returns:
            pd.DataFrame: Dataframe with ratings (user_id, item_id, rating, timestamp).
        """
        _, ext = os.path.splitext(file_path)

        if ext == ".dat":
            df = pd.read_csv(
                file_path,
                sep="::",
                engine="python",  # нужно для разделителя из нескольких символов
                names=["user_id", "item_id", "rating", "timestamp"],
            )
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            # Check if the CSV has headers; if not, set manually
            if set(["user_id", "item_id", "rating", "timestamp"]).issubset(df.columns):
                pass  # headers are already present
            else:
                df.columns = ["user_id", "item_id", "rating", "timestamp"]
        else:
            raise ValueError(
                f"Формат файла {ext} не поддерживается. Используйте .dat или .csv."
            )
        self.ratings = df
        print(self.ratings)

    def clean_data(self):
        # Ensure correct data types
        self.ratings["user_id"] = self.ratings["user_id"].astype(int)
        self.ratings["item_id"] = self.ratings["item_id"].astype(int)
        self.ratings["rating"] = self.ratings["rating"].astype(float)
        self.ratings["timestamp"] = pd.to_datetime(self.ratings["timestamp"], unit="s")
        print("Очистка данных завершена")

    def filter_data(self):
        # Filter users and items based on the minimum number of interactions
        user_counts = self.ratings["user_id"].value_counts()
        item_counts = self.ratings["item_id"].value_counts()

        valid_users = user_counts[user_counts >= self.min_user_interactions].index
        valid_items = item_counts[item_counts >= self.min_item_interactions].index

        before = len(self.ratings)
        self.ratings = self.ratings[
            self.ratings["user_id"].isin(valid_users)
            & self.ratings["item_id"].isin(valid_items)
        ]
        after = len(self.ratings)
        print(f"Фильтрация завершена: удалено {before - after} записей")

    def binarize_interactions(self):
        # Convert all ratings into binary interactions
        self.ratings["interaction"] = 1
        print("Бинаризация завершена")

    def process(self):
        self.load_data(file_path="dataset_ml-1m/ratings.dat")
        self.clean_data()
        self.filter_data()
        self.binarize_interactions()
        return self.ratings

    def split_by_days(self, output_dir: str, window_size_days: int = 5, last_days: int = 30):
        """
        Split data into daily subsets: for each day save both current day data and training data
        over the past `window_size_days` days.

        Args:
            output_dir (str): Directory where the subsets will be saved.
            window_size_days (int): Size of the window in days.
            last_days (int): Number of last days to process.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        min_date = self.ratings["timestamp"].min().normalize()
        max_date = self.ratings["timestamp"].max().normalize()

        current_date = max_date - timedelta(days=last_days) + timedelta(days=window_size_days)

        while current_date <= max_date:
            # Define the period: current day
            current_day_mask = self.ratings["timestamp"].dt.date == current_date.date()
            current_day_subset = self.ratings.loc[current_day_mask]

            # # Define the period: current day
            date_folder = os.path.join(output_dir, current_date.strftime("%Y-%m-%d"))
            os.makedirs(date_folder, exist_ok=True)
            current_day_subset.to_csv(os.path.join(date_folder, "ratings.csv"), index=False)

            # Save training data for the last N days before the current day
            training_start_date = current_date - timedelta(days=window_size_days)
            training_mask = (
                self.ratings["timestamp"].dt.date >= training_start_date.date()
            ) & (self.ratings["timestamp"].dt.date < current_date.date())
            training_subset = self.ratings.loc[training_mask]

            # Log the saved data
            training_subset.to_csv(os.path.join(date_folder, "training_data.csv"), index=False)
            print(f"Сохранено {len(current_day_subset)} записей в папку {date_folder} для текущего дня и {len(training_subset)} записей в обучающий набор.")

            # Move to the next day
            current_date += timedelta(days=1)


if __name__ == "__main__":
    processor = DataProcessor()
    interactions_df = processor.process()
    processor.split_by_days(output_dir="batches")
