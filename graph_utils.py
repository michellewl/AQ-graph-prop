import numpy as np
import scipy
import pandas as pd
import requests
from os import makedirs, path, listdir, remove
from bs4 import BeautifulSoup, SoupStrainer
import zipfile as zpf
from shutil import rmtree
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist, cosine
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import minimize
import matplotlib
from matplotlib import cm
from datetime import datetime

import httplib2
import geopandas as gpd
from tqdm import tqdm
from typing import Tuple, List, Optional, Union

# ------------------------ #
# Plot functions

def plot_on_map(data_geodataframe: gpd.GeoDataFrame, 
                map_geodataframe: gpd.GeoDataFrame, 
                data_column: str = None, 
                map_column: str = None, 
                data_cmap: str = None, 
                map_cmap: str = None, 
                data_color: str = None, 
                map_color: str = "whitesmoke", 
                data_markersize_factor: float = 0.01,  # Relative to fontsize
                map_edge_color: str = "black", 
                colorbar: bool = False, 
                title: str = "Greater London", 
                fontsize: int = 25, 
                figsize: tuple = (20, 10), 
                axis: str = "off",
                mark: bool = False,
                mark_column: str = "@SiteCode") -> None:
    """
    Plot data on a geographic map using GeoPandas and Matplotlib.

    Parameters:
    - data_geodataframe: GeoDataFrame containing data to be plotted.
    - map_geodataframe: GeoDataFrame containing the base map.
    - data_column: Column in data_geodataframe used for coloring data points.
    - map_column: Column in map_geodataframe used for coloring the base map.
    - data_cmap: Colormap for data points.
    - map_cmap: Colormap for the base map.
    - data_color: Color for data points.
    - map_color: Color for the base map.
    - data_markersize_factor: Size factor of markers for data points relative to fontsize.
    - map_edge_color: Edge color for the base map.
    - colorbar: Boolean indicating whether to display a colorbar.
    - title: Title of the plot.
    - fontsize: Font size for the title and labels.
    - figsize: Size of the figure (width, height).
    - axis: Matplotlib axis parameter.
    - mark: Boolean indicating whether to mark specific data points on the map.
    - mark_column: Column used to filter data for marking.

    Returns:
    - None
    """
    
    # Calculate marker size based on fontsize
    data_markersize = fontsize * data_markersize_factor
    
    # Plot the base map
    base = data_geodataframe.plot(column=data_column, 
                                  ax=map_geodataframe.plot(column=map_column, 
                                                           figsize=figsize, 
                                                           color=map_color, 
                                                           edgecolor=map_edge_color, 
                                                           cmap=map_cmap), 
                                  color=data_color, cmap=data_cmap, markersize=data_markersize)
    
    # Add colorbar if specified
    if colorbar:
        colorbar_max = data_geodataframe[data_column].max()
        norm = plt.Normalize(data_geodataframe[data_column].min(), colorbar_max)
        plt.colorbar(plt.cm.ScalarMappable(cmap=data_cmap, norm=norm)).set_label(data_column)
    
    # Mark specific points on the map if specified
    if mark:
        marked = data_geodataframe[data_geodataframe[mark_column] == mark]
        marked.plot(ax=base, marker='x', color='black', markersize=fontsize * 0.6)  # 0.6 is a relative size factor
    
    # Set plot title and axis labels
    plt.suptitle(title, fontsize=fontsize)
    plt.xlabel('Longitude', fontsize=fontsize * 0.7)  # 0.7 is a relative size factor
    plt.ylabel('Latitude', fontsize=fontsize * 0.7)  # 0.7 is a relative size factor
    
    # Set font sizes for ticks
    plt.xticks(fontsize=fontsize * 0.8)  # 0.8 is a relative size factor
    plt.yticks(fontsize=fontsize * 0.8)  # 0.8 is a relative size factor
    
    # Set axis parameter
    plt.axis(axis)
    
    # Display the plot
    plt.show()

# ------------------------ #
# Graph classes

class Dataset():
    def __init__(self, csv_file):
        # Initialize Dataset object with a CSV file
        self.df = pd.read_csv(csv_file)
        # Store the original DataFrame for reference
        self.orig = self.df.copy()
        # Convert the 'date' column to datetime format
        self.df['date'] = pd.to_datetime(self.df['date'])

    def drop_null(self, nan_percent):
        # Drop columns where the proportion of NaN elements exceeds nan_percent
        min_count = int(((100 - nan_percent) / 100) * self.df.shape[0] + 1)
        return self.df.dropna(axis=1, thresh=min_count)

    def fill_mean(self):
        # Fill NaN values with the mean of each column
        return self.df.fillna(self.df.mean())

    def group(self, freq):
        # Group the data by the specified frequency (month/year) and average across this period
        df = self.df.groupby(pd.Grouper(key="date", freq=freq)).mean()
        return df

    def group_and_fill(self, freq):
        # Group the data by the specified frequency (month/year), average across this period,
        # and fill NaN values using forward fill and backward fill
        df = self.df.groupby(pd.Grouper(key="date", freq=freq)).mean()
        return df.ffill().bfill()

    def fill(self):
        # Fill NaN values in each column by using the mean of the corresponding month and year
        df = self.df.copy()
        for col in df.columns.drop('date'):
            df[col] = df[col].fillna(df.groupby([df.date.dt.year, df.date.dt.month])[col].transform('mean'))
        return df.ffill().bfill()

class ComputeAM():
    def __init__(self, df):
        # Initialize ComputeAM object with a DataFrame
        am_shape = (df.shape[1], df.shape[1])
        # Create an adjacency matrix (AM) filled with zeros
        self.am = pd.DataFrame(np.zeros(shape=am_shape), columns=df.columns, index=df.columns)

    def euclidean_dist(self, df):
        # Calculate Euclidean distance between columns in the DataFrame
        dist_arr = squareform(pdist(df.transpose()))
        return pd.DataFrame(dist_arr, columns=df.columns.unique(), index=df.columns.unique())

    def cosine_dist(self, df):
        # Calculate cosine similarity between columns in the DataFrame
        dist_arr = cosine_similarity(df.transpose())
        np.fill_diagonal(dist_arr, 0)
        return pd.DataFrame(dist_arr, columns=df.columns.unique(), index=df.columns.unique())

    def threshold_euclidean(self, df, threshold):
        # Threshold Euclidean distances in the DataFrame
        df[df >= threshold] = 0
        df[df < threshold] = 1
        np.fill_diagonal(df.values, 0)
        return df

    def diagonal_degree(self, df):
        # Calculate diagonal degree matrix based on the sum of edges for each node
        diag_series = np.diag(df.sum())
        degree_mat = pd.DataFrame(diag_series, columns=df.columns.unique(), index=df.columns.unique())
        return degree_mat
    
class GraphPropagation:
    def __init__(self):
        pass
    
    def threshold_am(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Apply thresholding to the adjacency matrix.

        Parameters:
        - df: DataFrame representing the adjacency matrix.
        - threshold: Threshold value for binary thresholding.

        Returns:
        - DataFrame with binary values after applying the threshold.
        """
        result = df.copy()
        for col in result.columns:
            result[col] = np.where(result[col] >= threshold, 1, 0)
        np.fill_diagonal(result.values, 1)
        return result
    
    def diagonal_degree(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute diagonal degree matrix from the adjacency matrix.

        Parameters:
        - df: DataFrame representing the adjacency matrix.

        Returns:
        - DataFrame representing the diagonal degree matrix.
        """
        diag_series = np.diag(df.sum())
        result = pd.DataFrame(diag_series, columns=df.columns.unique(), index=df.columns.unique())
        return result

# ------------------------ #
# Graph functions

# Graph propagation algorithm

def D_pow(mat: np.ndarray, power: float) -> np.ndarray:
    """
    Compute the fractional matrix power of a matrix.

    Parameters:
    - mat: Input matrix.
    - power: Power to which the matrix is raised.

    Returns:
    - Resulting matrix after applying the fractional matrix power.
    """
    return scipy.linalg.fractional_matrix_power(mat, power)

def basic_graph_propagation(X: np.ndarray, A: np.ndarray, w: list, L: int, a: float = 0.5, b: float = 0.5) -> np.ndarray:
    """
    Basic graph propagation algorithm.

    Parameters:
    - X: Input data.
    - A: Adjacency matrix.
    - w: List of weights.
    - L: Number of iterations.
    - a: Parameter for D matrix power in the numerator (default is 0.5).
    - b: Parameter for D matrix power in the denominator (default is 0.5).

    Returns:
    - Resulting matrix after graph propagation.
    """
    D_list = np.sum(A, axis=1)  # D matrix
    w = np.array(w)
    prop_matrix = np.diag(D_list**-a).dot(A).dot(np.diag(D_list**-b))  # DAD^(-1)
    prop_matrix = np.nan_to_num(prop_matrix)  # Convert NaNs to 0s

    pi = np.zeros_like(X)
    r = X
    for i in range(L):
        Y_i = w[i:].sum()
        Y_iplus = w[i+1:].sum()

        # Update pi estimate
        q = (w[i]/Y_i) * r
        pi += q

        # Update r
        r = (Y_i/Y_iplus) * prop_matrix.dot(r.T).T

    q = w[L]/w[L:].sum() * r
    pi += q
    return pi

def fill_and_refactor(gap_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fill gaps in the input DataFrame and compute a refactored matrix based on Euclidean distance.

    Parameters:
    - gap_data: DataFrame containing data with gaps.

    Returns:
    - Tuple containing:
        - DataFrame with filled gaps using forward-fill and backward-fill.
        - Refactored matrix based on Euclidean distance.
    """
    filled_data = gap_data.ffill().bfill()  # Forward-fill and backward-fill to fill gaps

    adjacency_matrix = ComputeAM(filled_data) # Compute adjacency matrix
    euclidean_am = adjacency_matrix.euclidean_dist(filled_data)  # Compute Euclidean distance-based adjacency matrix

    mean = euclidean_am.mean().mean()  # Compute the mean of the Euclidean distance matrix
    refactored = (mean / euclidean_am)
    np.fill_diagonal(refactored.values, 0)  # Set diagonal elements to 0

    return filled_data, refactored

def euclidean_AM(filled_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Euclidean distance-based adjacency matrix.

    Parameters:
    - filled_data: DataFrame with filled gaps.

    Returns:
    - Euclidean distance-based adjacency matrix.
    """
    adjacency_matrix = ComputeAM(filled_data) # Compute adjacency matrix
    euclidean_am = adjacency_matrix.euclidean_dist(filled_data) # Compute Euclidean distance-based adjacency matrix

    mean = euclidean_am.mean().mean()  # Compute the mean of the Euclidean distance matrix
    refactored = (mean / euclidean_am)
    np.fill_diagonal(refactored.values, 0)  # Set diagonal elements to 0
    
    return refactored

def get_L(matrix: np.ndarray) -> int:
    """
    Compute the smallest positive integer L (number of hops or graph iterations) for effective graph propagation.
    The condition is that after raising the adjacency matrix to the power of L, the total number of non-zero elements
    becomes equal to the total number of elements in the matrix.

    Parameters:
    - matrix: Adjacency matrix representing the graph structure.

    Returns:
    - Smallest positive integer L (number of hops) satisfying the condition for effective graph propagation.
    """
    total = np.zeros_like(matrix)

    i = 0
    # Continue looping until the total number of non-zero elements equals the total number of elements
    while np.count_nonzero(total) != matrix.size:
        i += 1
        total += np.linalg.matrix_power(matrix, i)

        # Break the loop if i reaches 10 (arbitrary limit to avoid infinite loop)
        if i == 10:
            break

    return i

def compute_propagation_matrix(data: pd.DataFrame, euclideans: pd.DataFrame, threshold: float,
                                L: Optional[int] = None, alpha: Optional[float] = None,
                                w: np.ndarray = np.array([1, 0, 0, 0])) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute the graph propagation matrix and adjacency matrix.

    Parameters:
    - data: DataFrame containing input data.
    - euclideans: DataFrame representing Euclidean distances.
    - threshold: Threshold value for binary thresholding in adjacency matrix computation.
    - L: Number of hops or graph iterations (optional, default is determined using get_L function).
    - alpha: Parameter for weight computation (optional, default is None).
    - w: Array of weights for graph propagation (default is [1, 0, 0, 0]).

    Returns:
    - Tuple containing:
        - Graph propagation matrix Z.
        - Adjacency matrix A.
    """
    prop = GraphPropagation()
    
    # Compute adjacency matrix using thresholding
    A = prop.threshold_am(euclideans, threshold)

    # Adjust weights based on alpha if provided
    if alpha:
        w = [alpha * (1 - alpha)**i for i in range(10)]

    # Determine the number of hops or graph iterations if not provided
    if not L:
        L = get_L(A)

    # Convert data to numpy array and apply graph propagation algorithm
    array_data = data.to_numpy()
    Z = basic_graph_propagation(array_data, A, w, L)

    return Z, A



# ------------------------ #
# Hyperparameter optimisation with a complete subset of data

def get_complete_subset(df: pd.DataFrame, num_valid_values: int = 500) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract a complete subset from a DataFrame with missing values.

    Parameters:
    - df: DataFrame containing data.
    - num_valid_values: Number of valid values required for a column to be considered (default is 500).

    Returns:
    - Tuple containing the complete subset DataFrame and a list of column names.
    """
    # Initialize variables to track the maximum size and index
    max_size = 0
    max_index = 0

    # Loop through the DataFrame by rows, starting from index 0, with a step size of 5
    for i in range(0, df.shape[0], 5):
        # Create a boolean DataFrame indicating whether each element is null
        test = df.iloc[i:].isnull()

        # Reset the index of the boolean DataFrame and drop the old index
        test.reset_index(drop=True, inplace=True)

        # Find the index of the first occurrence of True (indicating missing value) in each column
        res = test.eq(True).idxmax()

        # Count the number of True values in each column and filter columns with count > num_valid_values
        size = res[res > num_valid_values].size

        # Update max_size and max_index if a larger valid column count is found
        if size > max_size:
            max_size = size
            max_index = i

    # Repeating the same process for the identified max_index
    test = df.iloc[max_index:].isnull()
    test.reset_index(drop=True, inplace=True)
    res = test.eq(True).idxmax()

    # Get the column names with valid values count > num_valid_values
    max_cols = res[res > num_valid_values].keys()

    # Extract the subset from the DataFrame using the identified columns and rows
    subset = df[max_cols].iloc[max_index:max_index + num_valid_values]

    # Return the complete subset and the column names
    return subset, max_cols

def get_custom_subset(df: pd.DataFrame, station_names: List[str], num_valid_values: int = 500) -> pd.DataFrame:
    """
    Extract a custom subset from a DataFrame with missing values.

    Parameters:
    - df: DataFrame containing data.
    - station_names: List of station (column) names to include in the subset.
    - num_valid_values: Number of valid values required for a column to be considered (default is 500).

    Returns:
    - DataFrame containing the custom subset.
    """
    # Initialize variables to track the maximum size and index
    max_size = 0
    max_index = 0

    # Loop through the DataFrame by rows, starting from index 0, with a step size of 5
    for i in range(0, df.shape[0], 5):
        # Create a boolean DataFrame indicating whether each element is null
        test = df.iloc[i:].isnull()

        # Reset the index of the boolean DataFrame and drop the old index
        test.reset_index(drop=True, inplace=True)

        # Find the index of the first occurrence of True (indicating missing value) in each column
        res = test.eq(True).idxmax()

        # Count the number of True values in each column and filter columns with count > num_valid_values
        size = res[res > num_valid_values].size

        # Update max_size and max_index if a larger valid column count is found
        if size > max_size:
            max_size = size
            max_index = i

    # Repeating the same process for the identified max_index
    test = df.iloc[max_index:].isnull()
    test.reset_index(drop=True, inplace=True)
    res = test.eq(True).idxmax()

    # Get the column names with valid values count > num_valid_values
    max_cols = res[res > num_valid_values].keys()

    # Extract the subset from the DataFrame using the identified columns and rows
    subset = df[station_names].iloc[max_index:max_index + num_valid_values]

    return subset

# def introduce_gaps(complete_subset: pd.DataFrame, num_missing_entries: int = 2000, seed: int = 0) -> Tuple[List[Tuple[int, int]], List[pd.Series], pd.DataFrame]:
#     """
#     Introduce gaps (replace random entries with NaNs) in a complete subset DataFrame.

#     Parameters:
#     - complete_subset: DataFrame representing the complete subset.
#     - num_missing_entries: Number of missing entries to be introduced (default is 2000).
#     - seed: Seed for NumPy random number generator (default is 0).

#     Returns:
#     - Tuple containing:
#         - List of tuples representing the indices of the introduced gaps.
#         - List of Series representing the initial true values before introducing gaps.
#         - DataFrame with introduced gaps.
#     """
#     np.random.seed(seed)
#     subset = complete_subset.copy()

#     # Replace random entries with NaNs
#     num_entries = complete_subset.size  # Total number of entries in the subset (e.g., 14000 for a 500x28 DataFrame)
#     missing_indices = np.random.choice(np.arange(num_entries), num_missing_entries, replace=False)
#     missing_entries = [(num // complete_subset.shape[1], num % complete_subset.shape[1]) for num in missing_indices]

#     # Store initial true values before introducing gaps
#     true_values = []
#     for entry in missing_entries:
#         true_values.append(subset.iloc[entry])
#         subset.iloc[entry] = np.nan

#     # Return information about introduced gaps, initial true values, and the modified subset
#     return missing_entries, true_values, subset

# # Introduce gaps of varying length

# def introduce_gaps_consecutive(complete_subset: pd.DataFrame, num_missing_periods: int = 100, max_period_length: int = 10, seed: int = 0) -> Tuple[List[List[Tuple[int, int]]], List[pd.Series], pd.DataFrame]:
#     """
#     Introduce gaps (replace consecutive entries with NaNs) in a complete subset DataFrame.

#     Parameters:
#     - complete_subset: DataFrame representing the complete subset.
#     - num_missing_periods: Number of missing data periods to be introduced (default is 100).
#     - max_period_length: Maximum length of missing data periods (default is 10).
#     - seed: Seed for NumPy random number generator (default is 0).

#     Returns:
#     - Tuple containing:
#         - List of lists, where each sublist contains tuples representing the indices of the introduced gaps within a period.
#         - List of Series representing the initial true values before introducing gaps.
#         - DataFrame with introduced gaps.
#     """
#     np.random.seed(seed)
#     subset = complete_subset.copy()

#     # Replace consecutive entries with NaNs and store indices for each period
#     num_rows, num_cols = complete_subset.shape
#     missing_periods_indices = []

#     for _ in range(num_missing_periods):
#         start_row = np.random.randint(0, num_rows - max_period_length + 1)
#         start_col = np.random.randint(0, num_cols)
#         period_length = np.random.randint(1, max_period_length + 1)

#         missing_indices_period = [(start_row + i, start_col) for i in range(period_length)]
#         missing_periods_indices.append(missing_indices_period)

#         for row, col in missing_indices_period:
#             subset.iloc[row, col] = np.nan

#     # Store initial true values before introducing gaps
#     true_values = [complete_subset.iloc[row, col] for period in missing_periods_indices for row, col in period]

#     # Return information about introduced gaps, initial true values, and the modified subset
#     return missing_periods_indices, true_values, subset

def introduce_gaps(complete_subset: pd.DataFrame, proportion: float = 0.2, max_period_length: int = 1, species: str = None, plot_histogram: bool = False, seed: int = 0) -> Tuple[List[List[Tuple[int, int]]], List[pd.Series], pd.DataFrame]:
    """
    Introduce gaps (replace single or consecutive entries with NaNs) in a complete subset DataFrame.

    Parameters:
    - complete_subset: DataFrame representing the complete subset.
    - proportion: Proportion of missing data to be introduced (default is 0.2).
    - max_period_length: Maximum length of missing data periods (default is 10).
    - species: Name of the species for which gaps are introduced (default is 'PM10').
    - seed: Seed for NumPy random number generator (default is 0).

    Returns:
    - Tuple containing:
        - List of lists, where each sublist contains tuples representing the indices of the introduced gaps within a period.
        - List of Series representing the initial true values before introducing gaps.
        - DataFrame with introduced gaps.
    """
    np.random.seed(seed)
    subset = complete_subset.copy()

    # Calculate number of missing periods based on proportion and max_period_length
    num_missing_periods = int(complete_subset.size * proportion) // max_period_length

    # Replace consecutive entries with NaNs and store indices for each period
    num_rows, num_cols = complete_subset.shape
    missing_periods_indices = []
    for _ in range(num_missing_periods):
        # Select random column if species is not specified; otherwise, select species-specific column
        if species is None:
            start_col = np.random.randint(0, num_cols)
        else:
            column_names = complete_subset.columns
            species_indices = [i for i, name in enumerate(column_names) if name.endswith(f'_{species}')]
            start_col = np.random.randint(min(species_indices), max(species_indices) + 1)
        start_row = np.random.randint(0, num_rows - max_period_length + 1)
        period_length = np.random.randint(1, max_period_length + 1)

        missing_indices_period = [(start_row + i, start_col) for i in range(period_length)]
        missing_periods_indices.append(missing_indices_period)

        # Replace values with NaNs for introduced gaps
        for row, col in missing_indices_period:
            subset.iloc[row, col] = np.nan

    # Store initial true values before introducing gaps
    true_values = [complete_subset.iloc[row, col] for period in missing_periods_indices for row, col in period]

    if plot_histogram:
        # Calculate lengths of missing data periods
        missing_periods_lengths = [len(period) for period in missing_periods_indices]
        # Plot histogram
        plt.hist(missing_periods_lengths, bins=range(1, max(missing_periods_lengths) + 1), align='left')
        plt.xlabel('Length of missing data period (hours)')
        plt.ylabel('Frequency')
        if species is None:
            plt.title('Histogram of artificially introduced missing data periods')
        else:
            plt.title(f'Histogram of artificially introduced missing data periods ({species})')
        plt.show()

    # Flatten the missing_periods_indices list of lists into a single list of tuples (this is necessary for downstream code)
    missing_periods_indices = [index for sublist in missing_periods_indices for index in sublist]

    # Return information about introduced gaps, initial true values, and the modified subset
    return missing_periods_indices, true_values, subset




# ------------------------ #
# Compute error metrics

def rmse_error(initial, final):
    """
    Calculate the Root Mean Squared Error (RMSE) between initial and final values.

    Parameters:
    - initial: List of initial values.
    - final: List of final (predicted) values.

    Returns:
    - RMSE value.
    """
    return np.linalg.norm(np.array(initial) - np.array(final)) / len(initial)**0.5

def absolute_error(initial, final):
    """
    Calculate the Mean Absolute Error (MAE) between initial and final values.

    Parameters:
    - initial: List of initial values.
    - final: List of final (predicted) values.

    Returns:
    - MAE value.
    """
    return np.mean(np.absolute(np.array(initial) - np.array(final)))

def smape_error(initial, final):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) between initial and final values.

    Parameters:
    - initial: List of initial values.
    - final: List of final (predicted) values.

    Returns:
    - SMAPE value.
    """
    initial, final = np.array(initial), np.array(final)
    num = np.absolute(initial - final)
    den = (np.absolute(initial) + np.absolute(final)) / 2
    elems = num/den
    return (np.sum(elems) / elems.size)*100

def compute_error(alpha: float, threshold: float, L: Union[float, int], 
                   initial: List[float], nan_entries: List[int], 
                   data: pd.DataFrame, euclideans: pd.DataFrame, 
                   error_type: str = 'rmse') -> float:
    """
    Compute error for the graph propagation algorithm with variable parameters.

    Parameters:
    - alpha: Weight parameter for each hop.
    - threshold: Threshold value for binary thresholding in adjacency matrix computation.
    - L: Number of hops or graph iterations.
    - initial: List of initial values.
    - nan_entries: Indices of entries with forced gaps.
    - data: Original dataset.
    - euclideans: Euclidean distance matrix.
    - error_type: Type of error to compute ('rmse' or 'absolute').

    Returns:
    - Computed error value.
    """
    prop = GraphPropagation()
    A = prop.threshold_am(euclideans, threshold)
    w = [alpha * (1 - alpha)**i for i in range(10)]

    # Apply graph propagation algorithm
    array_data = data.to_numpy()
    Z = basic_graph_propagation(array_data, A, w, int(round(L)))

    final = [Z[entry] for entry in nan_entries]

    if error_type == 'rmse':
        error = rmse_error(initial, final)
    elif error_type == 'absolute':
        error = absolute_error(initial, final)
    elif error_type == 'smape':
        error = smape_error(initial, final)
    else:
        raise ValueError("Invalid error_type. Supported types: 'rmse', 'absolute', 'smape'.")

    return float(error)  # Ensure the return value is a scalar

# ------------------------ #

# Normalise data between 0 and 1

def normalise_data(data):
    mins = data.min()
    maxs = data.max()
    normalized_data = (data - mins) / (maxs - mins)
    return normalized_data, mins, maxs

def denormalise_data(normalized_data, mins, maxs):
    return normalized_data * (maxs - mins) + mins