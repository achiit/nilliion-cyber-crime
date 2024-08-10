from matplotlib import pyplot as plt
from seaborn import histplot as sns_histplot
from pandas import DataFrame
import math


def plot_distributions(
    train_data: DataFrame, num_columns: int, row_height: int, size: int, fig_name: str
) -> None:
    # Plot train data in one frame
    # Calculate the number of rows and columns for subplots
    num_plots = len(train_data.columns)
    # num_cols = 7  # Number of columns for subplots
    num_rows = (num_plots - 1) // num_columns + 1  # Calculate the number of rows needed

    # Used to modify desired height of the frame
    # row_height = 2  # Adjust this value as needed
    fig_height = row_height * num_rows

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(size, fig_height))

    # Plot distribution of each attribute
    for i, column in enumerate(train_data.columns):
        row_index = i // num_columns
        col_index = i % num_columns
        sns_histplot(train_data[column], kde=True, ax=axes[row_index, col_index])
        axes[row_index, col_index].set_title(column)
        axes[row_index, col_index].set_xlabel("")
        axes[row_index, col_index].set_ylabel("Frequency")

    # Adjust layout
    fig.suptitle(fig_name)
    plt.tight_layout()
    plt.show()

    return None


def calc_scaling_factor(data: any, precision: int) -> int:
    # Calc max absolute value
    max_abs_value = max(abs(value) for value in data)

    # Calc scaling factor
    scaling_factor = 10 ** (precision - math.ceil(math.log10(max_abs_value)))
    return scaling_factor


def compute_scaled_data(data: any, scaling_factor: int) -> list[int]:
    # Scale values
    scaled_data = [round((value) * scaling_factor) for value in data]
    return scaled_data


def compute_prediction(test_data: list[int], theta: list[int]) -> int:
    # Perform element-wise multiplication and summation for test data
    prediction = 0
    for x_val, theta_val in zip(test_data, theta):
        prediction += x_val * theta_val
    return prediction
