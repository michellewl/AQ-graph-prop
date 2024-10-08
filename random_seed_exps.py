import numpy as np
import pandas as pd
# import requests
from os import makedirs, path, listdir, remove
# from bs4 import BeautifulSoup, SoupStrainer
import zipfile as zpf
from shutil import rmtree
import matplotlib.pyplot as plt
# import matplotlib
# from matplotlib import cm
# from datetime import datetime
# from tqdm import tqdm
# from typing import Tuple, List, Optional, Union

from graph_utils import *


species = "NO2"
region = "London"
start_date = "1996-01-01"
end_date = "2021-01-01"
data_folder = "/Users/michellewan/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/MEng_Kevin/Data and code"
data_filename = f"LAQN_{species}_{start_date}_{end_date}.csv"

data = Dataset(path.join(data_folder, data_filename))

hourly_df = data.df.set_index('date', inplace=False)
print(hourly_df.shape)

# Get the training set (2011 to 2019)
complete_subset_train, column_names_train = get_complete_subset(
    hourly_df.loc[(hourly_df.index >= '2011-01-01') & 
                  (hourly_df.index < '2019-01-01')], 
    num_valid_values=500
)

# Get the test set (between the end of training set and 2020)
df_test = hourly_df.loc[(hourly_df.index > complete_subset_train.index.max()) & 
                        (hourly_df.index < '2020-01-01')]
complete_subset_test, column_names_test = get_complete_subset(df_test, num_valid_values=500)

# Filter the dataframe between 26 March 2020 and 1 June 2020
lockdown_df = hourly_df.loc[(hourly_df.index >= '2020-03-26') & (hourly_df.index <= '2020-06-01')]
complete_subset_covid, column_names_covid = get_complete_subset(lockdown_df, num_valid_values=500)


print("Complete training subset shape:", complete_subset_train.shape)
print("Stations:", column_names_train)
print(complete_subset_train.index.min(), complete_subset_train.index.max())
train_set = complete_subset_train.copy()

print("Complete test subset shape:", complete_subset_test.shape)
print("Stations:", column_names_test)
print(complete_subset_test.index.min(), complete_subset_test.index.max())
test_set = complete_subset_test.copy()

print("Complete test_covid subset shape:", complete_subset_covid.shape)
print("Stations:", column_names_covid)
print(complete_subset_covid.index.min(), complete_subset_covid.index.max())
test_set_covid = complete_subset_covid.copy() # Use all of the lockdown data as the test set



# Prepare DataFrames to store the results
df_missing_proportions = pd.DataFrame(columns=['Random_Seed', 'Train_Missing%', 'Test_Missing%', 'COVID_Missing%'])
df_tuned_hyperparameters = pd.DataFrame(columns=['Random_Seed', 'Alpha', 'L_hops', 'Threshold'])
df_smape_test = pd.DataFrame(columns=['Random_Seed', 'Station_Annual_Mean', 'Poly_Order_1', 'Poly_Order_2', 'Poly_Order_3', 'Graph_Propagation'])
df_smape_covid = pd.DataFrame(columns=['Random_Seed', 'Station_Annual_Mean', 'Poly_Order_1', 'Poly_Order_2', 'Poly_Order_3', 'Graph_Propagation'])

# Define random seeds to iterate over
random_seeds = range(1, 11)  # Example: seeds from 1 to 10

# Main loop to iterate over random seeds
for random_seed in random_seeds:

    # Introduce consecutive gaps with the seed
    gap_proportion = 0.21
    max_gap_length = 20

    # Call the introduce_gaps function for training, test, and COVID sets
    gap_indices_train, true_values_train, subset_train = introduce_gaps(train_set, proportion=gap_proportion, max_period_length=max_gap_length, seed=random_seed)
    gap_indices_test, true_values_test, subset_test = introduce_gaps(test_set, proportion=gap_proportion, max_period_length=max_gap_length, seed=random_seed)
    gap_indices_test_covid, true_values_test_covid, subset_test_covid = introduce_gaps(test_set_covid, proportion=gap_proportion, max_period_length=max_gap_length, seed=random_seed)

    # Compute missing data proportions
    train_missing_pct = (subset_train.isna().sum().sum() / subset_train.size) * 100
    test_missing_pct = (subset_test.isna().sum().sum() / subset_test.size) * 100
    covid_missing_pct = (subset_test_covid.isna().sum().sum() / subset_test_covid.size) * 100

    # Add missing data proportions to the DataFrame
    df_missing_proportions = df_missing_proportions.append({'Random_Seed': random_seed, 
                                                            'Train_Missing%': train_missing_pct, 
                                                            'Test_Missing%': test_missing_pct,
                                                            'COVID_Missing%': covid_missing_pct}, ignore_index=True)

    # Perform the same imputation methods and error calculations as before (Station Annual Mean, Polynomial Orders, Graph Propagation)
    # Store the SMAPE results for each imputation method for both the test and COVID sets

    smape_errors_test = []
    smape_errors_covid = []

    # Baseline: Station Annual Mean
    filled_data_test = subset_test.groupby(subset_test.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))
    filled_data_test.reset_index(level='year', drop=True, inplace=True)
    estimated_values_test = [filled_data_test.iloc[idx[0], idx[1]] for idx in gap_indices_test]
    smape_test = smape_error(true_values_test, estimated_values_test)
    smape_errors_test.append(smape_test)

    filled_data_covid = subset_test_covid.groupby(subset_test_covid.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))
    filled_data_covid.reset_index(level='year', drop=True, inplace=True)
    estimated_values_covid = [filled_data_covid.iloc[idx[0], idx[1]] for idx in gap_indices_test_covid]
    smape_covid = smape_error(true_values_test_covid, estimated_values_covid)
    smape_errors_covid.append(smape_covid)

    # Polynomial Interpolation
    for order in range(1, 4):
        filled_data_test = subset_test.interpolate(method='polynomial', order=order, axis=0)
        filled_data_covid = subset_test_covid.interpolate(method='polynomial', order=order, axis=0)
        
        estimated_values_test = [filled_data_test.iloc[idx[0], idx[1]] for idx in gap_indices_test]
        smape_test = smape_error(true_values_test, estimated_values_test)
        smape_errors_test.append(smape_test)

        estimated_values_covid = [filled_data_covid.iloc[idx[0], idx[1]] for idx in gap_indices_test_covid]
        smape_covid = smape_error(true_values_test_covid, estimated_values_covid)
        smape_errors_covid.append(smape_covid)

    # Graph Propagation
    # Initialise training dataset with estimated interpolated values
    order = 1
    filled_data = subset_train.interpolate(method='polynomial', order=order, axis=0)

    # Check if there are still NaN values after interpolation
    remaining_gaps = filled_data.isna().sum().sum()

    if remaining_gaps > 0:
        print(f"Applying fill with annual mean concentrations for {remaining_gaps} remaining NaN values.")
        
        # Group by year and fill NaN values with the mean of each group
        filled_data = filled_data.groupby(filled_data.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))

        # Drop the additional "year" index created by groupby
        filled_data.reset_index(level='year', drop=True, inplace=True)

    euclidean = euclidean_AM(filled_data)
    # print(euclidean)

    # Graph propagation hyperparameters

    initialise_alpha = 0.2218
    initialise_hop = 2
    initialise_threshold = 1.06

    # Plotting for different parameters

    # Plotting for different alpha values
    plt.figure(1)
    alpha_err = []
    alpha_range = np.linspace(0.0, 0.6, 101)
    for alpha in alpha_range:
        # Compute error for each alpha value
        err = compute_error(alpha, initialise_threshold, initialise_hop, true_values_train, gap_indices_train, filled_data, euclidean)
        alpha_err.append(err)
    plt.plot(alpha_range, alpha_err)
    plt.title('RMSE Error vs. Alpha', fontsize=18)
    plt.xlabel('Alpha', fontsize=14)
    plt.ylabel('Error ($\mu g/m^3$)', fontsize=14)

    # Plotting for different hop values
    plt.figure(2)
    hop_err = []
    hop_range = np.arange(1, 6)
    for L in hop_range:
        # Compute error for each number of hops
        err = compute_error(initialise_alpha, initialise_threshold, L, true_values_train, gap_indices_train, filled_data, euclidean)
        hop_err.append(err)
    plt.plot(hop_range, hop_err)
    plt.title('RMSE Error vs. Number of Hops', fontsize=18)
    plt.xlabel('Hops', fontsize=14)
    plt.ylabel('Error ($\mu g/m^3$)', fontsize=14)

    # Plotting for different threshold values
    plt.figure(3)
    threshold_err = []
    threshold_range = np.linspace(1.0, 2.0, 101)
    for threshold in threshold_range:
        # Compute error for each threshold value
        err = compute_error(initialise_alpha, threshold, initialise_hop, true_values_train, gap_indices_train, filled_data, euclidean)
        threshold_err.append(err)
    plt.plot(threshold_range, threshold_err)
    plt.title('RMSE Error vs. Threshold', fontsize=18)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Error ($\mu g/m^3$)', fontsize=14)

    # Assign tuned hyperparameters
    alpha_err = np.nan_to_num(alpha_err, nan=np.inf)
    hop_err = np.nan_to_num(hop_err, nan=np.inf)
    threshold_err = np.nan_to_num(threshold_err, nan=np.inf)

    tuned_alpha = alpha_range[np.argmin(alpha_err)]
    tuned_hop = hop_range[np.argmin(hop_err)]
    tuned_threshold = threshold_range[np.argmin(threshold_err)]

    # COVID
    # Initialise test dataset with estimated interpolated values
    order = 1
    filled_data_covid = subset_test_covid.interpolate(method='polynomial', order=order, axis=0)

    # Check if there are still NaN values after interpolation
    remaining_gaps = filled_data_covid.isna().sum().sum()

    if remaining_gaps > 0:
        
        # Group by year and fill NaN values with the mean of each group
        filled_data_covid = filled_data_covid.groupby(filled_data_covid.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))

        # Drop the additional "year" index created by groupby
        filled_data_covid.reset_index(level='year', drop=True, inplace=True)

    # Use the tuned hyperparameters from the pre-COVID training set

    # Compute euclidean distance matrix    
    euclidean_covid = euclidean_AM(filled_data_covid)
    
    # Save tuned hyperparameters to the DataFrame
    df_tuned_hyperparameters = df_tuned_hyperparameters.append({'Random_Seed': random_seed, 
                                                                'Alpha': tuned_alpha, 
                                                                'L_hops': tuned_hop,
                                                                'Threshold': tuned_threshold}, ignore_index=True)

    # Graph Propagation Test Set
    Z_test, A_test = compute_propagation_matrix(filled_data_test, euclidean, threshold=tuned_threshold, L=tuned_hop, alpha=tuned_alpha)
    propagated_values_test = [Z_test[entry] for entry in gap_indices_test]
    smape_test_gp = smape_error(true_values_test, propagated_values_test)
    smape_errors_test.append(smape_test_gp)

    # Graph Propagation COVID Set
    Z_covid, A_covid = compute_propagation_matrix(filled_data_covid, euclidean_covid, threshold=tuned_threshold, L=tuned_hop, alpha=tuned_alpha)
    propagated_values_covid = [Z_covid[entry] for entry in gap_indices_test_covid]
    smape_covid_gp = smape_error(true_values_test_covid, propagated_values_covid)
    smape_errors_covid.append(smape_covid_gp)

    # Save SMAPE results to DataFrames
    df_smape_test = df_smape_test.append({'Random_Seed': random_seed, 
                                          'Station_Annual_Mean': smape_errors_test[0], 
                                          'Poly_Order_1': smape_errors_test[1], 
                                          'Poly_Order_2': smape_errors_test[2], 
                                          'Poly_Order_3': smape_errors_test[3], 
                                          'Graph_Propagation': smape_errors_test[4]}, ignore_index=True)

    df_smape_covid = df_smape_covid.append({'Random_Seed': random_seed, 
                                            'Station_Annual_Mean': smape_errors_covid[0], 
                                            'Poly_Order_1': smape_errors_covid[1], 
                                            'Poly_Order_2': smape_errors_covid[2], 
                                            'Poly_Order_3': smape_errors_covid[3], 
                                            'Graph_Propagation': smape_errors_covid[4]}, ignore_index=True)

    # Save figures instead of showing them
    plt.figure(1)
    plt.plot(alpha_range, alpha_err)
    plt.title('RMSE Error vs. Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Error ($\mu g/m^3$)')
    plt.savefig(f'alpha_vs_rmse_seed_{random_seed}.png')
    plt.close()

    plt.figure(2)
    plt.plot(hop_range, hop_err)
    plt.title('RMSE Error vs. Number of Hops')
    plt.xlabel('Hops')
    plt.ylabel('Error ($\mu g/m^3$)')
    plt.savefig(f'hop_vs_rmse_seed_{random_seed}.png')
    plt.close()

    plt.figure(3)
    plt.plot(threshold_range, threshold_err)
    plt.title('RMSE Error vs. Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Error ($\mu g/m^3$)')
    plt.savefig(f'threshold_vs_rmse_seed_{random_seed}.png')
    plt.close()

# Save DataFrames to CSV files
df_missing_proportions.to_csv('missing_data_proportions.csv', index=False)
df_tuned_hyperparameters.to_csv('tuned_hyperparameters.csv', index=False)
df_smape_test.to_csv('smape_test_results.csv', index=False)
df_smape_covid.to_csv('smape_covid_results.csv', index=False)

print("All results saved successfully.")


# # INTRODUCE CONSECUTIVE GAPS

# gap_proportion = 0.21 # Proportion of missing entries
# max_gap_length = 20 # Maximum length of missing data periods
# random_seed = 9


# # Call the introduce_gaps function
# gap_indices_train, true_values_train, subset_train = introduce_gaps(train_set, proportion=gap_proportion, max_period_length=max_gap_length, seed=random_seed, plot_histogram=True)
# gap_indices_test, true_values_test, subset_test = introduce_gaps(test_set, proportion=gap_proportion, max_period_length=max_gap_length, seed=random_seed, plot_histogram=True)

# print(f"Proportion of missing data (training): {(subset_train.isna().sum().sum()/(subset_train.size))*100:.3f}%")
# print(f"Proportion of missing data (test): {(subset_test.isna().sum().sum()/(subset_test.size))*100:.3f}%")


# # Group by year and fill NaN values with the mean of each group
# filled_data = subset_test.groupby(subset_test.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))

# # Drop the additional "year" index created by groupby
# filled_data.reset_index(level='year', drop=True, inplace=True)

# # Extract only the estimated values for NaN indices
# estimated_values = [filled_data.iloc[idx[0], idx[1]] for idx in gap_indices_test]

# # Compute error
# print(f"RMSE test set error after filling {species} with station annual means: {rmse_error(true_values_test, estimated_values):.2f}")
# print(f"SMAPE test set error after filling {species} with station annual means: {smape_error(true_values_test, estimated_values):.2f}%")

# print("Test set errors after filling with polynomial interpolation:")

# for order in range(1, 4):
#     filled_data = subset_test.interpolate(method='polynomial', order=order, axis=0)

#     # Check if there are still NaN values after interpolation
#     remaining_gaps = filled_data.isna().sum().sum()

#     if remaining_gaps > 0:
#         print(f"Applying fill with annual mean concentrations for {remaining_gaps} remaining NaN values.")
        
#         # Group by year and fill NaN values with the mean of each group
#         filled_data = filled_data.groupby(filled_data.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))

#         # Drop the additional "year" index created by groupby
#         filled_data.reset_index(level='year', drop=True, inplace=True)

#     # Extract only the estimated values for NaN indices
#     estimated_values = [filled_data.iloc[idx[0], idx[1]] for idx in gap_indices_test]

#     # Compute error for each method and print the results
#     print(f"Order {order} RMSE: {rmse_error(true_values_test, estimated_values):.2f}")
#     print(f"Order {order} SMAPE: {smape_error(true_values_test, estimated_values):.2f}%")



# # Initialise training dataset with estimated interpolated values
# order = 1
# filled_data = subset_train.interpolate(method='polynomial', order=order, axis=0)

# # Check if there are still NaN values after interpolation
# remaining_gaps = filled_data.isna().sum().sum()

# if remaining_gaps > 0:
#     print(f"Applying fill with annual mean concentrations for {remaining_gaps} remaining NaN values.")
    
#     # Group by year and fill NaN values with the mean of each group
#     filled_data = filled_data.groupby(filled_data.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))

#     # Drop the additional "year" index created by groupby
#     filled_data.reset_index(level='year', drop=True, inplace=True)

# euclidean = euclidean_AM(filled_data)
# # print(euclidean)

# # Graph propagation hyperparameters

# initialise_alpha = 0.2218
# initialise_hop = 2
# initialise_threshold = 1.06

# # Plotting for different parameters

# # Plotting for different alpha values
# plt.figure(1)
# alpha_err = []
# alpha_range = np.linspace(0.0, 0.6, 101)
# for alpha in alpha_range:
#     # Compute error for each alpha value
#     err = compute_error(alpha, initialise_threshold, initialise_hop, true_values_train, gap_indices_train, filled_data, euclidean)
#     alpha_err.append(err)
# plt.plot(alpha_range, alpha_err)
# plt.title('RMSE Error vs. Alpha', fontsize=18)
# plt.xlabel('Alpha', fontsize=14)
# plt.ylabel('Error ($\mu g/m^3$)', fontsize=14)

# # Plotting for different hop values
# plt.figure(2)
# hop_err = []
# hop_range = np.arange(1, 6)
# for L in hop_range:
#     # Compute error for each number of hops
#     err = compute_error(initialise_alpha, initialise_threshold, L, true_values_train, gap_indices_train, filled_data, euclidean)
#     hop_err.append(err)
# plt.plot(hop_range, hop_err)
# plt.title('RMSE Error vs. Number of Hops', fontsize=18)
# plt.xlabel('Hops', fontsize=14)
# plt.ylabel('Error ($\mu g/m^3$)', fontsize=14)

# # Plotting for different threshold values
# plt.figure(3)
# threshold_err = []
# threshold_range = np.linspace(1.0, 2.0, 101)
# for threshold in threshold_range:
#     # Compute error for each threshold value
#     err = compute_error(initialise_alpha, threshold, initialise_hop, true_values_train, gap_indices_train, filled_data, euclidean)
#     threshold_err.append(err)
# plt.plot(threshold_range, threshold_err)
# plt.title('RMSE Error vs. Threshold', fontsize=18)
# plt.xlabel('Threshold', fontsize=14)
# plt.ylabel('Error ($\mu g/m^3$)', fontsize=14)

# # Assign tuned hyperparameters
# alpha_err = np.nan_to_num(alpha_err, nan=np.inf)
# hop_err = np.nan_to_num(hop_err, nan=np.inf)
# threshold_err = np.nan_to_num(threshold_err, nan=np.inf)

# tuned_alpha = alpha_range[np.argmin(alpha_err)]
# tuned_hop = hop_range[np.argmin(hop_err)]
# tuned_threshold = threshold_range[np.argmin(threshold_err)]

# # Additional code to print minimum errors
# print('Minimum alpha error: ', min(alpha_err), 'Tuned alpha: ', tuned_alpha)
# print('Minimum L-hops error: ', min(hop_err), 'Tuned L-hops: ', tuned_hop)
# print('Minimum threshold error: ', min(threshold_err), 'Tuned threshold: ', tuned_threshold)

# tuned_alpha_hourly, tuned_hop_hourly, tuned_threshold_hourly = tuned_alpha, tuned_hop, tuned_threshold

# # Plot propagated training values against true training values

# Z, A = compute_propagation_matrix(filled_data, euclidean, threshold=tuned_threshold, L=tuned_hop, alpha=tuned_alpha)
# propagated_values = [Z[entry] for entry in gap_indices_train]

# x = np.arange(filled_data.max().max())
# plt.figure(figsize=(5, 5))
# plt.scatter(true_values_train, propagated_values)
# plt.plot(x, x, color='black', linestyle='--', label='1:1 Line')  # Adding a label for the 1:1 line
# # plt.title(f'Algorithm evaluation (train RMSE = {compute_error(tuned_alpha, tuned_threshold, tuned_hop, true_values_train, gap_indices_train, filled_data, euclidean):.2f})')
# plt.title(f"Training set")
# plt.xlabel(rf'True {species} concentration ($\mu g/m^3$)')  
# plt.ylabel(rf'Propagated {species} concentration ($\mu g/m^3$)')  
# plt.legend()  # Displaying the legend

# print(f"Training RMSE error: {compute_error(tuned_alpha, tuned_threshold, tuned_hop, true_values_train, gap_indices_train, filled_data, euclidean):.2f}")
# print(f"Training SMAPE error: {compute_error(tuned_alpha, tuned_threshold, tuned_hop, true_values_train, gap_indices_train, filled_data, euclidean, error_type='smape'):.2f}%")

# # Initialise test dataset with estimated interpolated values
# order = 1
# filled_data = subset_test.interpolate(method='polynomial', order=order, axis=0)

# # Check if there are still NaN values after interpolation
# remaining_gaps = filled_data.isna().sum().sum()

# if remaining_gaps > 0:
    
#     # Group by year and fill NaN values with the mean of each group
#     filled_data = filled_data.groupby(filled_data.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))

#     # Drop the additional "year" index created by groupby
#     filled_data.reset_index(level='year', drop=True, inplace=True)

# # Compute euclidean distance matrix
# euclidean = euclidean_AM(filled_data)
# # print(euclidean)

# # Plot propagated test values against true test values

# Z, A = compute_propagation_matrix(filled_data, euclidean, threshold=tuned_threshold, L=tuned_hop, alpha=tuned_alpha)
# propagated_values = [Z[entry] for entry in gap_indices_test]


# x = np.arange(filled_data.max().max())
# plt.figure(figsize=(5, 5))
# plt.scatter(true_values_test, propagated_values)
# plt.plot(x, x, color='black', linestyle='--', label='1:1 Line')  # Adding a label for the 1:1 line
# # plt.title(f'Algorithm evaluation (test RMSE = {compute_error(tuned_alpha, tuned_threshold, tuned_hop, true_values_test, gap_indices_test, filled_data, euclidean):.2f})')
# plt.title(f"Test set")
# plt.xlabel(rf'True {species} concentration ($\mu g/m^3$)')  
# plt.ylabel(rf'Propagated {species} concentration ($\mu g/m^3$)')  
# plt.legend()  # Displaying the legend

# print(f"Test RMSE error: {compute_error(tuned_alpha, tuned_threshold, tuned_hop, true_values_test, gap_indices_test, filled_data, euclidean):.2f}")
# print(f"Test SMAPE error: {compute_error(tuned_alpha, tuned_threshold, tuned_hop, true_values_test, gap_indices_test, filled_data, euclidean, error_type='smape'):.2f}%")

# column = column_names_test[0]

# plt.figure(figsize=(12, 6))

# if order > 1:
#     plt.plot(filled_data.index, filled_data[column], label=f"{column} polynomial-interpolated (order {order}) filled values", alpha=1, color="m")
# elif order == 1:
#     plt.plot(filled_data.index, filled_data[column], label=f"Linear model", alpha=1, color="m")

# # Replace graph-propagated values into the filled data from polynomial interpolation
# graph_prop = filled_data.copy()
# for index, value in zip(gap_indices_test, propagated_values):
#     graph_prop.iloc[index[0], index[1]] = value

# plt.plot(graph_prop.index, graph_prop[column], label=f"Graph propagation", alpha=1, color="tab:orange")

# # True values
# plt.plot(test_set.index, test_set[column], label="True values", alpha=1, color="tab:blue")


# # Add labels and legend
# plt.xlabel('Date')
# plt.ylabel(f'{species} concentration ($\mu g/m^3$)')
# plt.title(f'{column} original data vs filled data')
# plt.legend()
# plt.show()

# # COVID study

# print(species)
# print(hourly_df.shape)
# print(tuned_alpha_hourly, tuned_hop_hourly, tuned_threshold_hourly)

# print(gap_proportion, max_gap_length, random_seed)

# # COVID lockdown test set

# # Filter the dataframe between 26 March 2020 and 1 June 2020
# lockdown_df = hourly_df.loc[(hourly_df.index >= '2020-03-26') & (hourly_df.index <= '2020-06-01')]
# print(lockdown_df.shape)

# complete_subset_covid, column_names_covid = get_complete_subset(lockdown_df, num_valid_values=500)
# print("Complete subset shape:", complete_subset_covid.shape)
# print("Stations:", column_names_covid)
# print(complete_subset_covid.index.min(), complete_subset_covid.index.max())

# test_set_covid = complete_subset_covid.copy() # Use all of the lockdown data as the test set
# print("Test set shape:", test_set_covid.shape)

# gap_indices_test_covid, true_values_test_covid, subset_test_covid = introduce_gaps(test_set_covid, proportion=gap_proportion, max_period_length=max_gap_length, seed=random_seed, plot_histogram=True)

# print(f"Proportion of missing data (test): {(subset_test_covid.isna().sum().sum()/(subset_test_covid.size))*100:.3f}%")

# # BASELINE METHODS

# # Group by year and fill NaN values with the mean of each group
# filled_data_covid = subset_test_covid.groupby(subset_test_covid.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))
# filled_data_covid.reset_index(level='year', drop=True, inplace=True)
# estimated_values_covid = [filled_data_covid.iloc[idx[0], idx[1]] for idx in gap_indices_test_covid]
# # print(f"RMSE test set error after filling {species} with station annual means: {rmse_error(true_values_test_covid, estimated_values_covid):.2f}")
# print(f"SMAPE test set error after filling {species} with station annual means: {smape_error(true_values_test_covid, estimated_values_covid):.2f}%")

# # Linear and polynomial interpolation
# print("Test set errors after filling with polynomial interpolation:")
# for order in range(1, 4):
#     filled_data_covid = subset_test_covid.interpolate(method='polynomial', order=order, axis=0)
#     remaining_gaps_covid = filled_data_covid.isna().sum().sum()
#     if remaining_gaps_covid > 0:
#         print(f"Applying fill with annual mean concentrations for {remaining_gaps_covid} remaining NaN values.")
#         filled_data_covid = filled_data_covid.groupby(filled_data_covid.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))
#         filled_data_covid.reset_index(level='year', drop=True, inplace=True)
#     estimated_values_covid = [filled_data_covid.iloc[idx[0], idx[1]] for idx in gap_indices_test_covid]
#     # print(f"Order {order} RMSE: {rmse_error(true_values_test_covid, estimated_values_covid):.2f}")
#     print(f"Order {order} SMAPE: {smape_error(true_values_test_covid, estimated_values_covid):.2f}%")

# # GRAPH PROPAGATION

# # Initialise test dataset with estimated interpolated values
# order = 1
# filled_data_covid = subset_test_covid.interpolate(method='polynomial', order=order, axis=0)

# # Check if there are still NaN values after interpolation
# remaining_gaps = filled_data_covid.isna().sum().sum()

# if remaining_gaps > 0:
    
#     # Group by year and fill NaN values with the mean of each group
#     filled_data_covid = filled_data_covid.groupby(filled_data_covid.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))

#     # Drop the additional "year" index created by groupby
#     filled_data_covid.reset_index(level='year', drop=True, inplace=True)

# # Use the tuned hyperparameters from the pre-COVID training set

# print(f"Tuned hyperparameters from the pre-COVID training set: alpha = {tuned_alpha}, L-hops = {tuned_hop}, threshold = {tuned_threshold}")

# # Compute euclidean distance matrix    
# euclidean_covid = euclidean_AM(filled_data_covid)
# # print(euclidean)

# # Plot propagated test values against true test values
# print("Graph propagation test set errors")
# Z_covid, A_covid = compute_propagation_matrix(filled_data_covid, euclidean_covid, threshold=tuned_threshold, L=tuned_hop, alpha=tuned_alpha)
# propagated_values_covid = [Z_covid[entry] for entry in gap_indices_test_covid]

# x_covid = np.arange(filled_data_covid.max().max())
# plt.figure(figsize=(5, 5))
# plt.scatter(true_values_test_covid, propagated_values_covid)
# plt.plot(x_covid, x_covid, color='black', linestyle='--', label='1:1 Line')  # Adding a label for the 1:1 line
# # plt.title(f'Algorithm evaluation (test RMSE = {compute_error(tuned_alpha, tuned_threshold, tuned_hop, true_values_test_covid, gap_indices_test_covid, filled_data_covid, euclidean_covid):.2f})')
# plt.title(f"Test set (Low Emissions)")
# plt.xlabel(rf'True {species} concentration ($\mu g/m^3$)')  
# plt.ylabel(rf'Propagated {species} concentration ($\mu g/m^3$)')  
# plt.legend()  # Displaying the legend

# # print(f"RMSE error: {compute_error(tuned_alpha, tuned_threshold, tuned_hop, true_values_test_covid, gap_indices_test_covid, filled_data_covid, euclidean_covid):.2f}")
# print(f"SMAPE error: {compute_error(tuned_alpha, tuned_threshold, tuned_hop, true_values_test_covid, gap_indices_test_covid, filled_data_covid, euclidean_covid, error_type='smape'):.2f}%")

# # Plot timeseries
# column = column_names_covid[9]

# plt.figure(figsize=(12, 6))

# if order > 1:
#     plt.plot(filled_data_covid.index, filled_data_covid[column], label=f"{column} polynomial-interpolated (order {order}) filled values", alpha=1, color="m")
# elif order == 1:
#     plt.plot(filled_data_covid.index, filled_data_covid[column], label=f"Linear model", alpha=1, color="m")

# # Replace graph-propagated values into the filled data from polynomial interpolation
# graph_prop_covid = filled_data_covid.copy()
# for index, value in zip(gap_indices_test_covid, propagated_values_covid):
#     graph_prop_covid.iloc[index[0], index[1]] = value

# plt.plot(graph_prop_covid.index, graph_prop_covid[column], label=f"Graph propagation", alpha=1, color="tab:orange")

# # True values
# plt.plot(test_set_covid.index, test_set_covid[column], label="True values", alpha=1, color="tab:blue")


# # Add labels and legend
# plt.xlabel('Date')
# plt.ylabel(f'{species} concentration ($\mu g/m^3$)')
# plt.title(f'{column} original data vs filled data')
# plt.legend()
# plt.show()