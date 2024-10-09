import numpy as np
import pandas as pd
from os import makedirs, path
import matplotlib.pyplot as plt
from tqdm import tqdm
from graph_utils import *

species = "NO2"
region = "London"
start_date = "1996-01-01"
end_date = "2021-01-01"
data_folder = "/Users/michellewan/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/MEng_Kevin/Data and code"
data_filename = f"LAQN_{species}_{start_date}_{end_date}.csv"

# Create a folder for single species propagation
species_folder = f"single_{species}"
output_folder = path.join("results", species_folder)
makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

data = Dataset(path.join(data_folder, data_filename))

hourly_df = data.df.set_index('date', inplace=False)
print(f"Hourly dataframe shape: {hourly_df.shape}")

# Get the training set (2011 to 2019)
print("Obtaining training subset...")
complete_subset_train, column_names_train = get_complete_subset(
    hourly_df.loc[(hourly_df.index >= '2011-01-01') & 
                  (hourly_df.index < '2019-01-01')], 
    num_valid_values=500)
print("Complete training subset shape:", complete_subset_train.shape)
# print("Stations:", column_names_train)
print(f"Start: {complete_subset_train.index.min()} End: {complete_subset_train.index.max()}")
train_set = complete_subset_train.copy()
# Get the test set (between the end of training set and 2020)
print("Obtaining test subset...")
df_test = hourly_df.loc[(hourly_df.index > complete_subset_train.index.max()) & 
                        (hourly_df.index < '2020-01-01')]
complete_subset_test, column_names_test = get_complete_subset(df_test, num_valid_values=500)
print("Complete test subset shape:", complete_subset_test.shape)
# print("Stations:", column_names_test)
print(f"Start: {complete_subset_test.index.min()} End: {complete_subset_test.index.max()}")
test_set = complete_subset_test.copy()
# Get the COVID set between 26 March 2020 and 1 June 2020 (lockdown)
print("Obtaining COVID test subset...")
lockdown_df = hourly_df.loc[(hourly_df.index >= '2020-03-26') & (hourly_df.index <= '2020-06-01')]
complete_subset_covid, column_names_covid = get_complete_subset(lockdown_df, num_valid_values=500)
print("Complete test_covid subset shape:", complete_subset_covid.shape)
# print("Stations:", column_names_covid)
print(f"Start: {complete_subset_covid.index.min()} End: {complete_subset_covid.index.max()}")
test_set_covid = complete_subset_covid.copy() # Use all of the lockdown data as the test set

# Prepare DataFrames to store the results
df_missing_proportions = pd.DataFrame(columns=['random_seed', 'train_missing%', 'test_missing%', 'COVID_missing%'])
df_tuned_hyperparameters = pd.DataFrame(columns=['random_Seed', 'alpha', 'L_hops', 'threshold'])
df_smape_test = pd.DataFrame(columns=['random_seed', 'station_annual_mean', 'linear', 'poly_order_2', 'poly_order_3', 'graph_propagation'])
df_smape_covid = pd.DataFrame(columns=['random_seed', 'station_annual_mean', 'linear', 'poly_order_2', 'poly_order_3', 'graph_propagation'])

# Define random seeds to iterate over
random_seeds = range(0, 10)  

# Main loop to iterate over random seeds
for random_seed in tqdm(random_seeds, desc="Imputing with random seeds"):

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
    df_missing_proportions = pd.concat([df_missing_proportions, 
                                        pd.DataFrame({'random_seed': [random_seed], 
                                                      'train_missing%': [train_missing_pct], 
                                                      'test_missing%': [test_missing_pct],
                                                      'COVID_missing%': [covid_missing_pct]})], 
                                       ignore_index=True)

    smape_errors_test = []
    smape_errors_covid = []

    # Compute the necessary imputation methods and error calculations 
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
        # Check if there are still NaN values after interpolation
        remaining_gaps = filled_data_test.isna().sum().sum()
        if remaining_gaps > 0:
            filled_data_test = filled_data_test.groupby(filled_data_test.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))
            filled_data_test.reset_index(level='year', drop=True, inplace=True)
        estimated_values_test = [filled_data_test.iloc[idx[0], idx[1]] for idx in gap_indices_test]
        smape_test = smape_error(true_values_test, estimated_values_test)
        smape_errors_test.append(smape_test)

        filled_data_covid = subset_test_covid.interpolate(method='polynomial', order=order, axis=0)
        remaining_gaps = filled_data_covid.isna().sum().sum()
        if remaining_gaps > 0:
            filled_data_covid = filled_data_covid.groupby(filled_data_covid.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))
            filled_data_covid.reset_index(level='year', drop=True, inplace=True)
        estimated_values_covid = [filled_data_covid.iloc[idx[0], idx[1]] for idx in gap_indices_test_covid]
        smape_covid = smape_error(true_values_test_covid, estimated_values_covid)
        smape_errors_covid.append(smape_covid)

    # Graph Propagation
    # Initialise training dataset with estimated interpolated values
    order = 1
    filled_data_train = subset_train.interpolate(method='polynomial', order=order, axis=0)

    # Check if there are still NaN values after interpolation
    remaining_gaps = filled_data_train.isna().sum().sum()
    if remaining_gaps > 0:
        filled_data_train = filled_data_train.groupby(filled_data_train.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))
        filled_data_train.reset_index(level='year', drop=True, inplace=True)
    euclidean_train = euclidean_AM(filled_data_train)

    # Graph propagation hyperparameters
    initialise_alpha = 0.2218
    initialise_hop = 2
    initialise_threshold = 1.06
    # Hyperparameter tuning
    alpha_err = []
    alpha_range = np.linspace(0.0, 0.6, 101)
    for alpha in alpha_range:
        # Compute error for each alpha value
        err = compute_error(alpha, initialise_threshold, initialise_hop, true_values_train, gap_indices_train, filled_data_train, euclidean_train)
        alpha_err.append(err)
    hop_err = []
    hop_range = np.arange(1, 6)
    for L in hop_range:
        # Compute error for each number of hops
        err = compute_error(initialise_alpha, initialise_threshold, L, true_values_train, gap_indices_train, filled_data_train, euclidean_train)
        hop_err.append(err)
    threshold_err = []
    threshold_range = np.linspace(1.0, 2.0, 101)
    for threshold in threshold_range:
        # Compute error for each threshold value
        err = compute_error(initialise_alpha, threshold, initialise_hop, true_values_train, gap_indices_train, filled_data_train, euclidean_train)
        threshold_err.append(err)

    # Assign tuned hyperparameters
    alpha_err = np.nan_to_num(alpha_err, nan=np.inf)
    hop_err = np.nan_to_num(hop_err, nan=np.inf)
    threshold_err = np.nan_to_num(threshold_err, nan=np.inf)
    tuned_alpha = alpha_range[np.argmin(alpha_err)]
    tuned_hop = hop_range[np.argmin(hop_err)]
    tuned_threshold = threshold_range[np.argmin(threshold_err)]
    
    # Save tuned hyperparameters to the DataFrame
    df_tuned_hyperparameters = df_tuned_hyperparameters.append({'random_seed': random_seed, 
                                                                'alpha': tuned_alpha, 
                                                                'L_hops': tuned_hop,
                                                                'threshold': tuned_threshold}, ignore_index=True)

    # Initialise test datasets with estimated interpolated values
    filled_data_test = subset_test.interpolate(method='polynomial', order=1, axis=0)
    remaining_gaps = filled_data_test.isna().sum().sum()
    if remaining_gaps > 0:
        filled_data_test = filled_data_test.groupby(filled_data_test.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))
        filled_data_test.reset_index(level='year', drop=True, inplace=True)
    filled_data_covid = subset_test_covid.interpolate(method='polynomial', order=1, axis=0)
    remaining_gaps = filled_data_covid.isna().sum().sum()
    if remaining_gaps > 0:
        filled_data_covid = filled_data_covid.groupby(filled_data_covid.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))
        filled_data_covid.reset_index(level='year', drop=True, inplace=True)   
    
    # Compute euclidean distance matrix   
    euclidean_test = euclidean_AM(filled_data_test) 
    euclidean_covid = euclidean_AM(filled_data_covid)
    
    # Graph Propagation Test Set
    Z_test, A_test = compute_propagation_matrix(filled_data_test, euclidean_test, threshold=tuned_threshold, L=tuned_hop, alpha=tuned_alpha)
    propagated_values_test = [Z_test[entry] for entry in gap_indices_test]
    smape_test_gp = smape_error(true_values_test, propagated_values_test)
    smape_errors_test.append(smape_test_gp)

    # Graph Propagation COVID Set
    Z_covid, A_covid = compute_propagation_matrix(filled_data_covid, euclidean_covid, threshold=tuned_threshold, L=tuned_hop, alpha=tuned_alpha)
    propagated_values_covid = [Z_covid[entry] for entry in gap_indices_test_covid]
    smape_covid_gp = smape_error(true_values_test_covid, propagated_values_covid)
    smape_errors_covid.append(smape_covid_gp)


    # Save SMAPE results to DataFrames
    df_smape_test = pd.concat([df_smape_test, 
                               pd.DataFrame({'random_seed': [random_seed], 
                                             'station_annual_mean': [smape_errors_test[0]], 
                                             'linear': [smape_errors_test[1]], 
                                             'poly_order_2': [smape_errors_test[2]], 
                                             'poly_order_3': [smape_errors_test[3]], 
                                             'graph_propagation': [smape_errors_test[4]]})], 
                              ignore_index=True)

    df_smape_covid = pd.concat([df_smape_covid, 
                                pd.DataFrame({'random_seed': [random_seed], 
                                              'station_annual_mean': [smape_errors_covid[0]], 
                                              'linear': [smape_errors_covid[1]], 
                                              'poly_order_2': [smape_errors_covid[2]], 
                                              'poly_order_3': [smape_errors_covid[3]], 
                                              'graph_propagation': [smape_errors_covid[4]]})], 
                               ignore_index=True)

    # Save figures to output folder
    plt.figure(1)
    plt.plot(alpha_range, alpha_err)
    plt.title('RMSE vs alpha')
    plt.xlabel('alpha')
    plt.ylabel('error ($\mu g/m^3$)')
    plt.savefig(path.join(output_folder, f'alpha_vs_rmse_seed_{random_seed}.png'))
    plt.close()

    plt.figure(2)
    plt.plot(hop_range, hop_err)
    plt.title('RMSE vs number of hops')
    plt.xlabel('hops')
    plt.ylabel('error ($\mu g/m^3$)')
    plt.savefig(path.join(output_folder, f'hop_vs_rmse_seed_{random_seed}.png'))
    plt.close()

    plt.figure(3)
    plt.plot(threshold_range, threshold_err)
    plt.title('RMSE vs threshold')
    plt.xlabel('threshold')
    plt.ylabel('error ($\mu g/m^3$)')
    plt.savefig(path.join(output_folder, f'threshold_vs_rmse_seed_{random_seed}.png'))
    plt.close()

    # Save the scatter plots
    x = np.arange(filled_data_test.max().max())
    plt.figure(figsize=(6, 6))
    plt.scatter(true_values_test, propagated_values_test)
    plt.plot(x, x, color='black', linestyle='--', label='1:1 Line')
    plt.title(f"test set")
    plt.xlabel(rf'true {species} concentration ($\mu g/m^3$)')  
    plt.ylabel(rf'propagated {species} concentration ($\mu g/m^3$)')  
    plt.legend()
    plt.savefig(path.join(output_folder, f'test_scatter_{random_seed}.png'))
    plt.close()

    # Time series plot for test set
    plt.figure(figsize=(12, 6))
    column = column_names_test[0]
    plt.plot(filled_data_test.index, filled_data_test[column], label=f"Linear model", alpha=1, color="m")
    # Replace graph-propagated values into the filled data from polynomial interpolation
    graph_prop = filled_data_test.copy()
    for index, value in zip(gap_indices_test, propagated_values_test):
        graph_prop.iloc[index[0], index[1]] = value
    plt.plot(graph_prop.index, graph_prop[column], label=f"Graph propagation", alpha=1, color="tab:orange")
    # True values
    plt.plot(test_set.index, test_set[column], label="True values", alpha=1, color="tab:blue")
    # Add labels and legend
    plt.xlabel('Date')
    plt.ylabel(f'{species} concentration ($\mu g/m^3$)')
    plt.title(f'{column} original data vs filled data')
    plt.legend()
    plt.savefig(path.join(output_folder, f'test_timeseries_{random_seed}.png'))
    plt.close()

    # Repeat for COVID test set
    x_covid = np.arange(filled_data_covid.max().max())
    plt.figure(figsize=(5, 5))
    plt.scatter(true_values_test_covid, propagated_values_covid)
    plt.plot(x_covid, x_covid, color='black', linestyle='--', label='1:1 Line')
    plt.title(f"COVID test set")
    plt.xlabel(rf'true {species} concentration ($\mu g/m^3$)')  
    plt.ylabel(rf'propagated {species} concentration ($\mu g/m^3$)')  
    plt.legend()
    plt.savefig(path.join(output_folder, f'covid_scatter_{random_seed}.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    column = column_names_test[0]
    plt.plot(filled_data_test.index, filled_data_test[column], label=f"Linear model", alpha=1, color="m")
    # Replace graph-propagated values into the filled data from polynomial interpolation
    graph_prop = filled_data_test.copy()
    for index, value in zip(gap_indices_test, propagated_values_test):
        graph_prop.iloc[index[0], index[1]] = value
    plt.plot(graph_prop.index, graph_prop[column], label=f"Graph propagation", alpha=1, color="tab:orange")
    plt.plot(test_set.index, test_set[column], label="True values", alpha=1, color="tab:blue")
    plt.xlabel('Date')
    plt.ylabel(f'{species} concentration ($\mu g/m^3$)')
    plt.title(f'{column} original data vs filled data')
    plt.legend()
    plt.savefig(path.join(output_folder, f'covid_timeseries_{random_seed}.png'))
    plt.close()

# Save DataFrames to CSV files
print("Saving results...")
df_missing_proportions.to_csv(path.join(output_folder, 'missing_proportions.csv'), index=False)
df_tuned_hyperparameters.to_csv(path.join(output_folder, 'tuned_hyperparameters.csv'), index=False)
df_smape_test.to_csv(path.join(output_folder, 'smape_test.csv'), index=False)
df_smape_covid.to_csv(path.join(output_folder, 'smape_covid.csv'), index=False)
print(f"Saved results to {output_folder}")