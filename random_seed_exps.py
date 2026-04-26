import numpy as np
import pandas as pd
from os import makedirs, path
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats as stats
from graph_utils import *

# propagation_approach = "single" # "multi"

# species = "NO2" # gaps will be introduced into this species

# Define random seeds to iterate over
random_seeds = [0] # range(0, 50) 

for propagation_approach in ["single", "multi"]:

    for species in ["NO2", "PM10"]:
        print(f"{propagation_approach} species: {species}")

        # Create a folder for imputation results
        species_folder = f"{propagation_approach}_{species}"
        output_folder = path.join("results", species_folder)
        makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

        # Load data
        region = "London"
        start_date = "1996-01-01"
        end_date = "2021-01-01"
        data_folder = "data/"

        dfs = []

        if propagation_approach == "multi" or (propagation_approach=="single" and species=="NO2"):
            # Load NO2 data
            data_filename = f"LAQN_NO2_{start_date}_{end_date}.csv"
            data = Dataset(path.join(data_folder, data_filename))
            hourly_NO2 = data.df.set_index('date', inplace=False)
            # Columns suffix
            hourly_NO2 = hourly_NO2.rename(columns={c: c+'_NO2' for c in hourly_NO2.columns})
            dfs.append(hourly_NO2)
            
        if propagation_approach == "multi" or (propagation_approach=="single" and species=="PM10"):
            # Load PM10 data
            data_filename = f"LAQN_PM10_{start_date}_{end_date}.csv"
            data = Dataset(path.join(data_folder, data_filename))
            hourly_PM10 = data.df.set_index('date', inplace=False)
            # Columns suffix
            hourly_PM10 = hourly_PM10.rename(columns={c: c+'_PM10' for c in hourly_PM10.columns})
            dfs.append(hourly_PM10)
        # Concatenate species dataframes
        hourly_df = pd.concat(dfs, axis=1)

        print(f"Hourly dataframe shape: {hourly_df.shape}")

        # Get the training set (2011 to 2019)
        print("Obtaining training subset...")
        complete_subset_train, column_names_train = get_complete_subset(
            hourly_df.loc[(hourly_df.index >= '2011-01-01') & 
                        (hourly_df.index < '2019-01-01')], 
            num_valid_values=500)
        print("Complete training subset shape:", complete_subset_train.shape)
        print("Stations:", column_names_train)
        print(f"Start: {complete_subset_train.index.min()} End: {complete_subset_train.index.max()}")

        # Get the test set (between the end of training set and 2020)
        print("Obtaining test subset...")
        df_test = hourly_df.loc[(hourly_df.index > complete_subset_train.index.max()) & 
                                (hourly_df.index < '2020-01-01')]
        complete_subset_test, column_names_test = get_complete_subset(df_test, num_valid_values=500)
        print("Complete test subset shape:", complete_subset_test.shape)
        print("Stations:", column_names_test)
        print(f"Start: {complete_subset_test.index.min()} End: {complete_subset_test.index.max()}")

        # Get the COVID set between 26 March 2020 and 1 June 2020 (lockdown)
        print("Obtaining COVID test subset...")
        lockdown_df = hourly_df.loc[(hourly_df.index >= '2020-03-26') & (hourly_df.index <= '2020-06-01')]
        complete_subset_covid, column_names_covid = get_complete_subset(lockdown_df, num_valid_values=500)
        print("Complete test_covid subset shape:", complete_subset_covid.shape)
        print("Stations:", column_names_covid)
        print(f"Start: {complete_subset_covid.index.min()} End: {complete_subset_covid.index.max()}")

        # Identify the index of the first column of the species of interest
        first_index_test = [i for i, s in enumerate(column_names_test) if species in s][0]
        first_index_covid = [i for i, s in enumerate(column_names_covid) if species in s][0]

        # Normalise the data across both species
        normalised_train, mins_train, maxs_train = normalise_data(complete_subset_train)
        normalised_test, mins_test, maxs_test = normalise_data(complete_subset_test)
        normalised_covid, mins_covid, maxs_covid = normalise_data(complete_subset_covid)
        print("Normalised train data shape:", normalised_train.shape)
        print("Normalised test data shape:", normalised_test.shape)
        print("Normalised COVID test data shape:", normalised_covid.shape)

        # Prepare DataFrames to store the results
        df_missing_proportions = pd.DataFrame(columns=['random_seed', 'train_missing%', 'test_missing%', 'COVID_missing%'])
        df_tuned_hyperparameters = pd.DataFrame(columns=['random_seed', 'alpha', 'L_hops', 'threshold'])
        df_smape_test = pd.DataFrame(columns=['random_seed', 'station_annual_mean', 'linear', 'poly_order_2', 'poly_order_3', 'graph_propagation'])
        df_smape_covid = pd.DataFrame(columns=['random_seed', 'station_annual_mean', 'linear', 'poly_order_2', 'poly_order_3', 'graph_propagation']) 
        df_metrics_test = pd.DataFrame(columns=['random_seed', 'method', 'SMAPE', 'R2', 'RMSE', 'MAE'])
        df_metrics_covid = pd.DataFrame(columns=['random_seed', 'method', 'SMAPE', 'R2', 'RMSE', 'MAE'])

        # Main loop to iterate over random seeds
        for random_seed in tqdm(random_seeds, desc="Imputing with random seeds"):

            # Introduce consecutive gaps with the seed
            gap_proportion = 0.21
            max_gap_length = 20

            # Call the introduce_gaps function for training, test, and COVID sets
            gap_indices_train, true_values_train, subset_train = introduce_gaps(normalised_train, species=species, proportion=gap_proportion, max_period_length=max_gap_length, seed=random_seed)
            gap_indices_test, true_values_test, subset_test = introduce_gaps(normalised_test, species=species, proportion=gap_proportion, max_period_length=max_gap_length, seed=random_seed)
            gap_indices_test_covid, true_values_test_covid, subset_test_covid = introduce_gaps(normalised_covid, species=species, proportion=gap_proportion, max_period_length=max_gap_length, seed=random_seed)

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
            r2, rmse, mae = compute_metrics(true_values_test, estimated_values_test)
            df_metrics_test = pd.concat([df_metrics_test, pd.DataFrame({
                'random_seed': [random_seed],
                'method': ['station_annual_mean'],
                'SMAPE': [smape_test],
                'R2': [r2],
                'RMSE': [rmse],
                'MAE': [mae]
            })], ignore_index=True)

            filled_data_covid = subset_test_covid.groupby(subset_test_covid.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))
            filled_data_covid.reset_index(level='year', drop=True, inplace=True)
            estimated_values_covid = [filled_data_covid.iloc[idx[0], idx[1]] for idx in gap_indices_test_covid]
            smape_covid = smape_error(true_values_test_covid, estimated_values_covid)
            smape_errors_covid.append(smape_covid)
            r2, rmse, mae = compute_metrics(true_values_test_covid, estimated_values_covid)
            df_metrics_covid = pd.concat([df_metrics_covid, pd.DataFrame({
                'random_seed': [random_seed],
                'method': ['station_annual_mean'],
                'SMAPE': [smape_covid],
                'R2': [r2],
                'RMSE': [rmse],
                'MAE': [mae]
            })], ignore_index=True)

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
                r2, rmse, mae = compute_metrics(true_values_test, estimated_values_test)
                df_metrics_test = pd.concat([df_metrics_test, pd.DataFrame({
                    'random_seed': [random_seed],
                    'method': [f'polynomial_{order}'],
                    'SMAPE': [smape_test],
                    'R2': [r2],
                    'RMSE': [rmse],
                    'MAE': [mae]
                })], ignore_index=True)

                filled_data_covid = subset_test_covid.interpolate(method='polynomial', order=order, axis=0)
                remaining_gaps = filled_data_covid.isna().sum().sum()
                if remaining_gaps > 0:
                    filled_data_covid = filled_data_covid.groupby(filled_data_covid.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))
                    filled_data_covid.reset_index(level='year', drop=True, inplace=True)
                estimated_values_covid = [filled_data_covid.iloc[idx[0], idx[1]] for idx in gap_indices_test_covid]
                smape_covid = smape_error(true_values_test_covid, estimated_values_covid)
                smape_errors_covid.append(smape_covid)
                r2, rmse, mae = compute_metrics(true_values_test_covid, estimated_values_covid)
                df_metrics_covid = pd.concat([df_metrics_covid, pd.DataFrame({
                    'random_seed': [random_seed],
                    'method': [f'polynomial_{order}'],
                    'SMAPE': [smape_covid],
                    'R2': [r2],
                    'RMSE': [rmse],
                    'MAE': [mae]
                })], ignore_index=True)

            # Graph Propagation
            # Initialise training dataset with estimated interpolated values
            order = 3 #TK
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
            alpha_range = np.linspace(1.e-6, 0.6, 101)
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

            # CORRECTIONS
            # Fixing hyperparameters

            tuned_alpha = 0.23
            tuned_hop = 2
            tuned_threshold = 1.1
            
            # Save tuned hyperparameters to the DataFrame
            df_tuned_hyperparameters = df_tuned_hyperparameters.append({'random_seed': random_seed, 
                                                                        'alpha': tuned_alpha, 
                                                                        'L_hops': tuned_hop,
                                                                        'threshold': tuned_threshold}, ignore_index=True)

            # Initialise test datasets with estimated interpolated values
            filled_data_test = subset_test.interpolate(method='polynomial', order=order, axis=0)
            remaining_gaps = filled_data_test.isna().sum().sum()
            if remaining_gaps > 0:
                filled_data_test = filled_data_test.groupby(filled_data_test.index.year.rename('year')).apply(lambda group: group.fillna(group.mean()))
                filled_data_test.reset_index(level='year', drop=True, inplace=True)
            filled_data_covid = subset_test_covid.interpolate(method='polynomial', order=order, axis=0)
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
            r2, rmse, mae = compute_metrics(true_values_test, propagated_values_test)
            df_metrics_test = pd.concat([df_metrics_test, pd.DataFrame({
                'random_seed': [random_seed],
                'method': ['graph_propagation'],
                'SMAPE':[smape_test_gp],
                'R2': [r2],
                'RMSE': [rmse],
                'MAE': [mae]
            })], ignore_index=True)

            # Graph Propagation COVID Set
            Z_covid, A_covid = compute_propagation_matrix(filled_data_covid, euclidean_covid, threshold=tuned_threshold, L=tuned_hop, alpha=tuned_alpha)
            propagated_values_covid = [Z_covid[entry] for entry in gap_indices_test_covid]
            smape_covid_gp = smape_error(true_values_test_covid, propagated_values_covid)
            smape_errors_covid.append(smape_covid_gp)
            r2, rmse, mae = compute_metrics(true_values_test_covid, propagated_values_covid)
            df_metrics_covid = pd.concat([df_metrics_covid, pd.DataFrame({
                'random_seed': [random_seed],
                'method': ['graph_propagation'],
                'SMAPE': [smape_covid_gp], 
                'R2': [r2],
                'RMSE': [rmse],
                'MAE': [mae]
            })], ignore_index=True)

            # print(f"SMAPE true_values_test, propagated_values_test: {smape_test_gp}")
            # print(f"SMAPE filled_data_test and complete_subset_test: {smape_error(complete_subset_test.to_numpy().flatten().tolist(), filled_data_test.to_numpy().flatten().tolist())}")
            # print(f"SMAPE denormalised filled_data_test and complete_subset_test: {smape_error(denormalise_data(complete_subset_test, mins_test, maxs_test).to_numpy().flatten().tolist(), denormalise_data(filled_data_test, mins_test, maxs_test).to_numpy().flatten().tolist())}")

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

            # Scatter plot - test set
            x = np.linspace(min(min(true_values_test), min(propagated_values_test)), 
                            max(max(true_values_test), max(propagated_values_test)), 100)
            plt.figure(figsize=(6, 6))
            plt.scatter(true_values_test, propagated_values_test)
            plt.plot(x, x, color='black', linestyle='--', label='1:1 Line')
            r2, _, _ = compute_metrics(true_values_test, propagated_values_test)
            plt.text(0.05, 0.95, f'$R^2 = {r2:.3f}$', transform=plt.gca().transAxes, verticalalignment='top')
            plt.title(f"test set")
            plt.xlabel(rf'true {species} concentration (normalised values)')  
            plt.ylabel(rf'propagated {species} concentration (normalised values)')  
            plt.legend()
            plt.savefig(path.join(output_folder, f'test_scatter_{random_seed}.png'))
            plt.close()

            # Scatter plot - COVID test set
            x_covid = np.linspace(min(min(true_values_test_covid), min(propagated_values_covid)), 
                                  max(max(true_values_test_covid), max(propagated_values_covid)), 100)
            plt.figure(figsize=(5, 5))
            plt.scatter(true_values_test_covid, propagated_values_covid)
            plt.plot(x_covid, x_covid, color='black', linestyle='--', label='1:1 Line')
            r2, _, _ = compute_metrics(true_values_test_covid, propagated_values_covid)
            plt.text(0.05, 0.95, f'$R^2 = {r2:.3f}$', transform=plt.gca().transAxes, verticalalignment='top')
            plt.title(f"COVID test set")
            plt.xlabel(rf'true {species} concentration (normalised values)')  
            plt.ylabel(rf'propagated {species} concentration (normalised values)')  
            plt.legend()
            plt.savefig(path.join(output_folder, f'covid_scatter_{random_seed}.png'))
            plt.close()

            # Undo data normalisation for timeseries plotting
            initial_est_test = denormalise_data(filled_data_test, mins_test, maxs_test)
            initial_est_covid = denormalise_data(filled_data_covid, mins_covid, maxs_covid)

            # Time series plot for test set
            plt.figure(figsize=(12, 6))
            column = column_names_test[first_index_test]
            plt.plot(initial_est_test.index, initial_est_test[column], label=f"Poly{order} model", alpha=1, color="m") #TK
            # Replace graph-propagated values into the filled data from initial interpolation estimate
            graph_test = filled_data_test.copy()
            for index, value in zip(gap_indices_test, propagated_values_test):
                graph_test.iloc[index[0], index[1]] = value
            graph_test = denormalise_data(graph_test, mins_test, maxs_test)
            plt.plot(graph_test.index, graph_test[column], label=f"Graph propagation", alpha=1, color="tab:orange")
            # True values
            plt.plot(complete_subset_test.index, complete_subset_test[column], label="True values", alpha=1, color="tab:blue")
            # Add labels and legend
            plt.xlabel('Date')
            plt.ylabel(f'{species} concentration ($\mu g/m^3$)')
            plt.title(f'{column} original data vs filled data')
            plt.legend()
            plt.savefig(path.join(output_folder, f'test_timeseries_{random_seed}.png'))
            plt.close()

            # Time series plot for COVID test set
            plt.figure(figsize=(12, 6))
            column = column_names_covid[first_index_covid]
            plt.plot(initial_est_covid.index, initial_est_covid[column], label=f"Poly{order} model", alpha=1, color="m") #TK
            # Replace graph-propagated values into the filled data from polynomial interpolation
            graph_covid = filled_data_covid.copy()
            for index, value in zip(gap_indices_test_covid, propagated_values_covid):
                graph_covid.iloc[index[0], index[1]] = value
            graph_covid = denormalise_data(graph_covid, mins_covid, maxs_covid)
            plt.plot(graph_covid.index, graph_covid[column], label=f"Graph propagation", alpha=1, color="tab:orange")
            plt.plot(complete_subset_covid.index, complete_subset_covid[column], label="True values", alpha=1, color="tab:blue")
            plt.xlabel('Date')
            plt.ylabel(f'{species} concentration ($\mu g/m^3$)')
            plt.title(f'{column} original data vs filled data')
            plt.legend()
            plt.savefig(path.join(output_folder, f'covid_timeseries_{random_seed}.png'))
            plt.close()

            # Quantile-quantile plots
            residuals_test = np.array(true_values_test) - np.array(propagated_values_test)
            plt.figure()
            stats.probplot(residuals_test, dist="norm", plot=plt)
            plt.title("Q-Q plot (Test residuals)")
            plt.savefig(path.join(output_folder, f'graph_prop_qq_test_{random_seed}.png'))
            plt.close()
            # COVID set
            residuals_covid = np.array(true_values_test_covid) - np.array(propagated_values_covid)
            plt.figure()
            stats.probplot(residuals_covid, dist="norm", plot=plt)
            plt.title("Q-Q plot (Test residuals)")
            plt.savefig(path.join(output_folder, f'graph_prop_qq_covid_{random_seed}.png'))
            plt.close()

            # Residuals histogram and scatter
            # Test set
            plt.figure()
            plt.hist(residuals_test, bins=50)
            plt.title("Residual distribution (Test)")
            plt.xlabel("Error")
            plt.ylabel("Frequency")
            plt.savefig(path.join(output_folder, f'graph_prop_residual_hist_test_{random_seed}.png'))
            plt.close()

            plt.figure()
            plt.scatter(propagated_values_test, residuals_test)
            plt.axhline(0, linestyle='--')
            plt.title("Residuals vs Predictions")
            plt.xlabel("Predicted")
            plt.ylabel("Residuals")
            plt.savefig(path.join(output_folder, f'graph_prop_residual_scatter_test_{random_seed}.png'))
            plt.close()

            # COVID set
            plt.figure()
            plt.hist(residuals_covid, bins=50)
            plt.title("Residual distribution (COVID)")
            plt.xlabel("Error")
            plt.ylabel("Frequency")
            plt.savefig(path.join(output_folder, f'graph_prop_residual_hist_covid_{random_seed}.png'))
            plt.close()
            
            plt.figure()
            plt.scatter(propagated_values_covid, residuals_covid)
            plt.axhline(0, linestyle='--')
            plt.title("Residuals vs Predictions")
            plt.xlabel("Predicted")
            plt.ylabel("Residuals")
            plt.savefig(path.join(output_folder, f'graph_prop_residual_scatter_covid_{random_seed}.png'))
            plt.close()

        # Save DataFrames to CSV files
        print("Saving results...")
        df_missing_proportions.to_csv(path.join(output_folder, 'missing_proportions.csv'), index=False)
        df_tuned_hyperparameters.to_csv(path.join(output_folder, 'tuned_hyperparameters.csv'), index=False)
        df_smape_test.to_csv(path.join(output_folder, 'smape_test.csv'), index=False)
        df_smape_covid.to_csv(path.join(output_folder, 'smape_covid.csv'), index=False)
        df_metrics_test.to_csv(path.join(output_folder, 'metrics_test.csv'), index=False)
        df_metrics_covid.to_csv(path.join(output_folder, 'metrics_covid.csv'), index=False)
        
        # Calculate mean and standard deviation of results
        print("Summarising results...")
        csv_files = [f for f in listdir(output_folder) if (f.endswith('.csv') and "summary" not in f)]

        # for file in csv_files:
        #     print("\n", file)
        #     filepath = path.join(output_folder, file)

        #     df = pd.read_csv(filepath, index_col=0, header=0)
        #     mean_df = df.mean()
        #     std_df = df.std()

        #     decimal_places = 2
        #     combined_df = mean_df.round(decimal_places).astype(str) + " ± " + std_df.round(decimal_places).astype(str)
        #     print(combined_df)

        #     combined_df.to_csv(path.join(output_folder, f"summary_{file}"), index=True)
        for file in csv_files:
            print("\n", file)
            filepath = path.join(output_folder, file)
            df = pd.read_csv(filepath)

            if "method" in df.columns:
                summary = df.groupby("method")[["SMAPE", "R2", "RMSE", "MAE"]].agg(["mean", "std"])

                # Format as "mean ± std"
                formatted = summary.copy()
                for col in ["SMAPE", "R2", "RMSE", "MAE"]:
                    formatted[(col, "mean")] = summary[(col, "mean")].round(2).astype(str)
                    formatted[(col, "std")] = summary[(col, "std")].round(2).astype(str)
                    formatted[col] = formatted[(col, "mean")] + " ± " + formatted[(col, "std")]

                # Keep only combined columns
                final_df = formatted[[col for col in ["SMAPE", "R2", "RMSE", "MAE"]]]

            else:
                # fallback for other CSVs (your existing behaviour)
                numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["random_seed"], errors="ignore")
                mean_df = numeric_df.mean()
                std_df = numeric_df.std()
                final_df = mean_df.round(2).astype(str) + " ± " + std_df.round(2).astype(str)

            final_df.to_csv(path.join(output_folder, f"summary_{file}"))

        print(f"Saved results to {output_folder}")

