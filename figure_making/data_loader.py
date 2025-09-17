import config
import pandas as pd
import numpy as np
import os
import pickle
import glob
from collections import defaultdict
import quality_control as qc


def load_session_dataframe(animal_id, df_name, session_id=None, session_long_name=None, file_format='parquet'):
    """
    Loads a session-specific DataFrame using an integer session index instead of the long identifier string.

    Args:
        animal_id (str): The ID of the animal (e.g., 'SZ036').
        df_name (str): The name of the data product to load (e.g., 'dFF0', 'zscore').
        session_id (int): The chronological index of the session (e.g., 0 for the first session).
        session_long_name (str): the session identifier that's formatted as 'yyyy-mm-ddThh-mm'.
        file_format (str): The file format ('parquet', 'csv', 'pickle').

    Returns:
        A pandas DataFrame, or None if the file is not found or an error occurs.
    """
    # 1. Construct the path to the animal's processed data directory using the config
    processed_dir = os.path.join(config.MAIN_DATA_ROOT, animal_id, config.PROCESSED_DATA_SUBDIR)
    if not os.path.exists(processed_dir):
        print(f"Warning: Processed directory not found at {processed_dir}")
        return None

    if session_long_name is not None:
        target_filename = f"{animal_id}_{session_long_name}_{df_name}.{file_format}"
    elif session_id is not None:
        # 2. Find all files in that directory that match the animal and data product name
        file_suffix = f"_{df_name}.{file_format}"

        # List all files in the directory and filter for the ones we want
        matching_files = [
            f for f in os.listdir(processed_dir)
            if f.startswith(f"{animal_id}_") and f.endswith(file_suffix)
        ]

        # 3. Sort the list of files chronologically (alphabetical sort works here)
        matching_files.sort()

        # 4. Check if the requested session_index is valid and select the target file
        if not 0 <= session_id < len(matching_files):
            print(f"Error: session_id '{session_id}' is out of bounds.")
            print(f"Found {len(matching_files)} sessions for '{animal_id}' with data product '{df_name}'.")
            # For debugging, see what files were found:
            # print("Found files:", matching_files)
            return None

        target_filename = matching_files[session_id]

    file_path = os.path.join(processed_dir, target_filename)

    print(f"Loading: {file_path}")  # Helpful for confirming the right file is being loaded
    print(f"session: {target_filename[:-21]}")
    session_long_name = target_filename[:-21]
    # 5. Load the file using the appropriate pandas function
    try:
        if file_format == 'parquet':
            return pd.read_parquet(file_path)
        elif file_format == 'csv':
            return pd.read_csv(file_path)
        elif file_format == 'pickle':
            return pd.read_pickle(file_path)
        else:
            print(f"Unsupported file format: {file_format}")
            return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_dataframes_for_animal_summary(animal_ids, df_name, day_0, hemisphere_qc=1, file_format='parquet'):
    """
    Loads and concatenates data files for a list of animals.

    Args:
        animal_ids (list): List of animal identifiers (strings).
        df_name (str): Common name identifier in the filenames (e.g., "DA_features").
        day_0 (str): The reference start date in 'YYYY-MM-DD' format.
        file_format (str, optional): The file extension ('parquet', 'csv', etc.). Defaults to 'parquet'.

    Returns:
        pandas.DataFrame: A single dataframe containing concatenated data from all animals,
                          or an empty DataFrame if no files were found.
    """

    # A list to store the individual dataframes before concatenation
    all_animal_data = []
    day_zero_datetime = pd.to_datetime(day_0)

    # Loop through each animal's data folder
    for animal in animal_ids:

        # Define the path to the processed data for the current animal
        processed_dir = os.path.join(config.MAIN_DATA_ROOT, animal, config.PROCESSED_DATA_SUBDIR)
        if not os.path.exists(processed_dir):
            print(f"Warning: Directory not found, skipping: {animal}")
            continue
        # Find all the files with the df_name and the file_format in the directory
        search_path = os.path.join(processed_dir, f"*_{df_name}.{file_format}")
        session_files = glob.glob(search_path)
        if not session_files:
            print(f"Info: No '{df_name}' files found for animal {animal} in {processed_dir}")
            continue

        animal_session_info = []
        for file in session_files:
            base_name = os.path.basename(file)
            temp_name = base_name.removeprefix(f"{animal}_")
            suffix_to_remove = f"_{df_name}.{file_format}"
            session_id = temp_name.removesuffix(suffix_to_remove)
            session_dt = pd.to_datetime(session_id, format='%Y-%m-%dT%H_%M')
            animal_session_info.append({'path': file, 'id': session_id, 'datetime': session_dt})

        daily_session_counter = defaultdict(int)
        animal_rules = qc.qc_selections[animal]
        for session_data in animal_session_info:
            session_dt = session_data['datetime']
            session_date_only = session_dt.date()
            day_relative = (session_date_only - day_zero_datetime.date()).days
            daily_session_counter[session_date_only] += 1
            session_of_day = daily_session_counter[session_date_only]

            if file_format == 'parquet':
                df = pd.read_parquet(session_data['path'])
            elif file_format == 'csv':
                df = pd.read_csv(session_data['path'])
            else:
                print(f"Unsupported file format: {file_format}")
                continue

            if hemisphere_qc:
                masks_to_keep = []
                for hemi in animal_rules:
                    masks_to_keep.append(df['hemisphere'] == hemi)
                if masks_to_keep:
                    final_mask = np.logical_or.reduce(masks_to_keep)
                    df = df[final_mask].reset_index(drop=True)

            if not df.empty:
                df['animal'] = animal
                df['session'] = session_data['id']
                df['day_relative'] = day_relative
                df['session_of_day'] = session_of_day
                all_animal_data.append(df)

    # Concatenate all the dataframes in the list into a single, master dataframe
    if not all_animal_data:
        print("Error: No dataframes were loaded. Returning an empty DataFrame.")
        return pd.DataFrame()
    else:
        master_dataframe = pd.concat(all_animal_data, ignore_index=True)
        print(master_dataframe.head())
        print(f"\nTotal number of data points: {len(master_dataframe)}")
        return master_dataframe


if __name__ == '__main__':
    # animal_str = 'SZ036'
    # session_id = 11
    # session_long_name = '2024-01-11T16_25'
    # zscore = load_session_dataframe(animal_str, 'zscore', session_long_name=session_long_name, session_id=11, file_format='parquet')

    animal_ids = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043"]
    master_dataframe = load_dataframes_for_animal_summary(animal_ids, 'DA_vs_features', day_0='2023-11-30',
                                                          file_format='parquet')

    print('Hello')
