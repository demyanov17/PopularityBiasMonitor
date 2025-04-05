import numpy as np
import pandas as pd
import logging as logger


class PopularityMetricsCalculator:
    def __init__(self, df_train: pd.DataFrame, key_column="group_id", ks=[1, 2, 5, 10]):
        """Calculate popularity metrics

        Information:
        1) https://arxiv.org/pdf/1901.07555.pdf
        2) https://browse.arxiv.org/pdf/2006.03715.pdf

        """

        # Column with item_id's or group_id's
        self.key_column = key_column

        # Calculate metrics for these top K
        self.ks = ks

        # Count of keys in train data
        self.df_train_key_count = df_train.groupby(key_column)["unique_id"].count().reset_index(name="count")

        # Number of unique_id's in train data
        self.num_unique_id_train = df_train["unique_id"].nunique()

    def _ARP(self, df_recs: pd.DataFrame, k: int, normalize: bool = False) -> float:
        """Average Recommendation Popularity"""

        # Keep only top K recommendations
        df_recs = df_recs.groupby("unique_id").head(k)

        # Add popularity to recommendations
        df_recs_popularity = df_recs.merge(self.df_train_key_count, on=self.key_column, how="left").fillna(0)

        # Normalize popularity over user count if necessary
        if normalize:
            df_recs_popularity["count"] = df_recs_popularity["count"] / self.num_unique_id_train

        return df_recs_popularity.groupby("unique_id")["count"].mean().mean()

    def _average_rank(self, df_recs: pd.DataFrame, k: int) -> float:
        """Average rank of recommendations"""

        # Keep only top K recommendations
        df_recs = df_recs.groupby("unique_id").head(k)

        # Sort so that most popular keys are at the top
        df_popularity = self.df_train_key_count.sort_values(by="count", ascending=False)

        # Add rank column
        df_popularity["rank"] = np.arange(len(df_popularity)) + 1

        return df_recs.merge(df_popularity[[self.key_column, "rank"]], on="group_id", how="inner")["rank"].mean()

    def _average_num_recs_per_user(self, df_recs: pd.DataFrame, k: int) -> float:
        """Average number of recommendations per user"""

        # Keep only top K recommendations
        df_recs = df_recs.groupby("unique_id").head(k)

        return len(df_recs) / df_recs["unique_id"].nunique()

    def _coverage(self, df_recs: pd.DataFrame, k: int) -> float:
        """Coverage of items in train dataset by recommendations"""

        # Keep only top K recommendations
        df_recs = df_recs.groupby("unique_id").head(k)

        # Unique items in train dataset
        items_train = self.df_train_key_count[self.key_column].unique()

        # Unique items in recommendations
        items_recs = df_recs[self.key_column].unique()

        return len(set(items_train) and set(items_recs)) / len(set(items_train))

    def _gini(self, df_recs: pd.DataFrame, k: int) -> float:
        """Gini index of recommendations"""

        # Keep only top K recommendations
        df_recs = df_recs.groupby("unique_id").head(k)

        # Calculate counts for all keys
        df_key_count = df_recs.groupby(self.key_column)["unique_id"].count().reset_index(name="count")

        # Get key probabilities and sort in ascending order
        key_probs_sorted = np.sort(np.asarray(df_key_count["count"] / len(df_recs)))

        # Number of keys (I in Gini formula)
        num_keys = len(key_probs_sorted)

        # Array with k's in Gini formula
        k_array = np.arange(num_keys) + 1

        return ((2 * k_array - num_keys - 1) * key_probs_sorted).sum() / num_keys

    def _entropy(self, df_recs: pd.DataFrame, k: int) -> float:
        """Entropy of recommendations"""

        # Keep only top recommendations
        df_recs = df_recs.groupby("unique_id").head(k)

        # Calculate counts for all keys
        df_key_count = df_recs.groupby(self.key_column)["unique_id"].count().reset_index(name="count")

        # Get key probabilities
        key_probs = np.asarray(df_key_count["count"] / len(df_recs))

        return -(key_probs * np.log(key_probs)).sum()

    def _APLT(self, df_recs: pd.DataFrame, k: int, tail_percentage: float = 0.8) -> float:
        """Average percentage of long tail items"""

        # Number of long tail keys out of all items in train
        num_tail_items = int(len(self.df_train_key_count) * tail_percentage)

        # Get tail keys from train
        tail_items = self.df_train_key_count.sort_values(by="count").head(num_tail_items)[self.key_column]

        # Keep only top recommendations
        df_recs = df_recs.groupby("unique_id").head(k)

        # Count number of recommendations per user
        df_recs_count = df_recs.groupby("unique_id")[self.key_column].count().reset_index(name="count")

        # Count number of tail recommendations per user
        df_tail_recs_count = (
            df_recs[df_recs[self.key_column].isin(tail_items)]
            .groupby("unique_id")[self.key_column]
            .count()
            .reset_index(name="tail_count")
        )

        # Single dataframe with all and tail recommendations count
        df_recs_count = df_recs_count.merge(df_tail_recs_count, on="unique_id", how="left").fillna(0)

        return sum(df_recs_count["tail_count"] / df_recs_count["count"]) / len(df_recs_count)

    def calculate(self, df_recs: pd.DataFrame) -> pd.DataFrame:
        """Main method to compute all metrics"""
        metrics_dict = {}
        for k in self.ks:
            metrics_dict[k] = {}
            metrics_dict[k]["APR"] = self._ARP(df_recs, k)
            metrics_dict[k]["APR_norm"] = self._ARP(df_recs, k, normalize=True)
            metrics_dict[k]["Avg_rank"] = self._average_rank(df_recs, k)
            metrics_dict[k]["Avg_num_recs"] = self._average_num_recs_per_user(df_recs, k)
            metrics_dict[k]["Coverage"] = self._coverage(df_recs, k)
            metrics_dict[k]["Gini"] = self._gini(df_recs, k)
            metrics_dict[k]["Entropy"] = self._entropy(df_recs, k)
            metrics_dict[k]["APLT"] = self._APLT(df_recs, k)
        df_popularity_metrics = pd.DataFrame(metrics_dict).T.reset_index().rename(columns={"index": "@k"})
        with pd.option_context(
            "display.max_columns",
            500,
            "display.width",
            1000,
            "display.max_rows",
            100,
            "display.float_format",
            lambda x: "%.5f" % x,
        ):
            logger.info(f"Popularity_metrics:\n{df_popularity_metrics}")
        return df_popularity_metrics
