import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from popularity_metrics import PopularityMetricsCalculator


class Visualizer:

    def __init__(self, json_data, output_dir="output_plots"):
        """
        Initialize the Visualizer class.

        Args:
            json_data (dict): Data loaded from a JSON file.
            output_dir (str): Directory to save the generated plots.
        """
        self.data = json_data
        self.output_dir = output_dir
        self.metrics = [
            "APR",
            "APR_norm",
            "Avg_rank",
            "Avg_num_recs",
            "Coverage",
            "Gini",
            "Entropy",
            "APLT",
        ]
        self.k_values = [1, 2, 5, 10]

        os.makedirs(self.output_dir, exist_ok=True)

    def plot_metric_trends(self, metric_name, k):
        """
        Plot the trend of a specific metric over time for a given k value.

        Args:
            metric_name (str): Name of the metric to plot (e.g., "APR").
            k (int): The k value for which the plot is generated.
        """

        metric_values = []
        dates = []

        # Collect metric values and corresponding dates
        for date, records in self.data.items():
            for record in records:
                if record["@k"] == k:
                    metric_values.append(record[metric_name])
                    dates.append(date)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(
            dates,
            metric_values,
            marker="o",
            label=f"{metric_name} for k={k}",
            color="b",
        )
        plt.xlabel("Date", fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f"{metric_name} Over Time (k={k})", fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save the plot as an SVG file for high quality
        plt.savefig(f"{self.output_dir}/{metric_name}_k{k}.pdf", format="pdf", dpi=300)
        plt.close()

    def plot_all_metrics(self):
        """
        Generate plots for all metrics and all k values.
        """
        base_dir = 'batches' # Specify the path to the folders with data
        output_json_path = 'popularity_metrics.json' # Specify the path to save the metrics JSON file

        popularity_calculator = PopularityMetricsCalculator()
        popularity_calculator.compute_on_batches(base_dir, output_json_path)
        print("Подсчёт метрик завершён. Графики отображены по соответствующему URL")

        for metric in self.metrics:
            for k in self.k_values:
                self.plot_metric_trends(metric, k)
