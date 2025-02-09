import pandas as pd
import os

def read_and_process_files(folder_path, prefix):
    metrics = ['Accuracy', 'F1_Score_Macro', 'F1_Score_Weighted']
    results_ideal = pd.DataFrame()
    results_self = pd.DataFrame()

    # List all files in the directory
    files = os.listdir(folder_path)

    # Process files for 'ideal' and 'self'
    for kind in ['ideal', 'self']:
        data_frames = []
        relevant_files = [file for file in files if file.startswith(f'experiment_results_{kind}_{prefix}')]

        for file_name in relevant_files:
            file_path = os.path.join(folder_path, file_name)
            # Read the file
            data = pd.read_csv(file_path)
            
            # Check if data is not empty and contains all required metrics
            if not data.empty and all(metric in data.columns for metric in metrics):
                data_frames.append(data[metrics])
        
        if data_frames:
            concatenated_data = pd.concat(data_frames)
            # Calculate the mean and standard deviation
            if kind == 'ideal':
                results_ideal = concatenated_data.agg(['mean', 'std'])
            else:
                results_self = concatenated_data.agg(['mean', 'std'])
        else:
            if kind == 'ideal':
                results_ideal = pd.DataFrame(columns=metrics, index=['mean', 'std'])
            else:
                results_self = pd.DataFrame(columns=metrics, index=['mean', 'std'])

    return results_ideal, results_self


def read_and_process_files_1(folder_path, prefix):
    metrics = ['Accuracy', 'F1_Score_Macro', 'F1_Score_Weighted']
    results_ideal = pd.DataFrame()
    results_self = pd.DataFrame()

    # List all files in the directory
    files = os.listdir(folder_path)

    # Process files for 'ideal' and 'self'
    for kind in ['ideal', 'self']:
        data_frames = []
        relevant_files = [file for file in files if file.startswith(f'experiment_results_{kind}_{prefix}')]

        for file_name in relevant_files:
            file_path = os.path.join(folder_path, file_name)
            # Read the file
            data = pd.read_csv(file_path)
            
            # Check if data is not empty and contains all required metrics
            if not data.empty and all(metric in data.columns for metric in metrics):
                data_frames.append(data[metrics])
        
        if data_frames:
            concatenated_data = pd.concat(data_frames)
            # Calculate the mean
            if kind == 'ideal':
                results_ideal = concatenated_data.mean()
            else:
                results_self = concatenated_data.mean()
        else:
            if kind == 'ideal':
                results_ideal = pd.Series(index=metrics)
            else:
                results_self = pd.Series(index=metrics)

    return results_ideal, results_self

# Define folder path
folder_path = 'results'
prefixes = ['BACK', 'LC', 'LLA', 'LT', 'LUA', 'RC', 'RLA', 'RT', 'RUA']

# Process files for each prefix and print the results
for prefix in prefixes:
    results_ideal, results_self = read_and_process_files(folder_path, prefix)
    print(f"Results Summary for {prefix} - Ideal vs Self:")
    print("Ideal:")
    print(results_ideal)
    print("Self:")
    print(results_self)
    print("\n")


# Define folder path and file prefixes
folder_path = 'results'
prefixes = ['BACK', 'LC', 'LLA', 'LT', 'LUA', 'RC', 'RLA', 'RT', 'RUA']

# Store results in a DataFrame
all_results_ideal = pd.DataFrame(columns=prefixes)
all_results_self = pd.DataFrame(columns=prefixes)

# Process files for each prefix and accumulate results
for prefix in prefixes:
    results_ideal, results_self = read_and_process_files_1(folder_path, prefix)
    all_results_ideal[prefix] = results_ideal
    all_results_self[prefix] = results_self

# Calculate the average performance drop for each metric
performance_drop = (all_results_ideal.mean(axis=1) - all_results_self.mean(axis=1)) / all_results_ideal.mean(axis=1) * 100

print("Average Performance Drop (%):")
print(performance_drop)