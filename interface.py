import os, json
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from collections import defaultdict
from visualization import Visualizer


class Board_Interface:
    def __init__(self, json_data):
        """
        Initialize the PopularityDashboard.

        Args:
            json_data (dict): Data loaded from a JSON file.
        """
        self.json_data = json_data
        self.plotter = Visualizer(json_data)  # Используем класс для построения графиков

    def run(self):
        """Run the Streamlit application"""
        st.title("Анализ метрик популярности рекомендаций")

        st.sidebar.header("Параметры отображения")
        metric_to_show = st.sidebar.selectbox(
            "Выберите метрику для отображения",
            options=[
                "APR",
                "APR_norm",
                "Avg_rank",
                "Avg_num_recs",
                "Coverage",
                "Gini",
                "Entropy",
                "APLT",
            ],
        )
        k_value = st.sidebar.selectbox("Выберите значение K", options=[1, 2, 5, 10])

        # Plot the selected metric
        st.write(f"### График изменения {metric_to_show} от дня для K={k_value}")
        self.plot_metric_streamlit(metric_to_show, k_value)

        # Display a brief summary
        st.write("### Краткий анализ метрик")
        self.display_summary()

    def plot_metric_streamlit(self, metric_name: str, k: int):
        """
        Plot the selected metric using Streamlit.

        Args:
            metric_name (str): Name of the metric to plot.
            k (int): Value of k for which to plot the metric.
        """

        metric_values = []
        dates = []

        # Collect metric values and corresponding dates
        for date, records in self.json_data.items():
            for record in records:
                if record["@k"] == k:
                    metric_values.append(record[metric_name])
                    dates.append(date)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            dates,
            metric_values,
            marker="o",
            label=f"{metric_name} for k={k}",
            color="b",
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f"{metric_name} Over Time (k={k})", fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(fig)

    def display_summary(self):
        """
        Display a brief textual summary of the metrics.
        """

        summary = defaultdict(str)
        for k in [1, 2, 5, 10]:
            summary["Coverage"] = f"Наилучшее покрытие при k={k}"
            summary["Gini"] = f"Наименьший индекс Джини при k={k}"
            summary["Entropy"] = f"Наибольшая энтропия при k={k}"

        # Отображение анализа
        st.markdown(f"- **Покрытие**: {summary['Coverage']}")
        st.markdown(f"- **Индекс Джини**: {summary['Gini']}")
        st.markdown(f"- **Энтропия**: {summary['Entropy']}")


if __name__ == "__main__":
    import sys

    # Load data from JSON
    with open("popularity_metrics.json", "r") as f:
        json_data = json.load(f)

    # Launch the dashboard
    dashboard = Board_Interface(json_data)
    dashboard.run()
