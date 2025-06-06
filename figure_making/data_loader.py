import config
import pandas as pd
import numpy as np
import os
import pickle


def load_session_dataframe(animal_id, session_id, df_name, file_format='parquet'):
    """
    Loads a session-specific DataFrame using an integer session index instead of the long identifier string.

    Args:
        animal_id (str): The ID of the animal (e.g., 'SZ036').
        session_id (int): The chronological index of the session (e.g., 0 for the first session).
        df_name (str): The name of the data product to load (e.g., 'dFF0', 'zscore').
        file_format (str): The file format ('parquet', 'csv', 'pickle').

    Returns:
        A pandas DataFrame, or None if the file is not found or an error occurs.
    """
    # 1. Construct the path to the animal's processed data directory using the config
    processed_dir = os.path.join(config.MAIN_DATA_ROOT, animal_id, config.PROCESSED_DATA_SUBDIR)
    if not os.path.exists(processed_dir):
        print(f"Warning: Processed directory not found at {processed_dir}")
        return None

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


def load_animal_summary_data(base_processed_dir, animal_id, summary_name, file_format='csv'):
    """Loads animal-level summary data."""
    # Similar logic, path might be slightly different (e.g., directly in animal_id/processed_data or a summary subfolder)
    # ...
    pass

if __name__ == '__main__':
    animal_str = 'SZ036'
    session_id = 11
    zscore = load_session_dataframe(animal_str, session_id, 'zscore', file_format='parquet')
    print('Hello')
