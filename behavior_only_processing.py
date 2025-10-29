import helper
import config
import pandas as pd
import os

def process_behavior_data(session_dir):
    pi_events = pd.read_csv(session_dir, na_values=['None'], skiprows=3)
    pi_events = helper.data_reduction(pi_events, lick_tol=.015, head_tol=0.2)
    pi_events = helper.add_2ndry_properties_to_pi_events(pi_events)
    pi_events.reset_index(drop=True, inplace=True)
    session_identifier = session_dir[-23:-4]
    return pi_events, session_identifier

def process_animal_data_and_save(animal_str):
    animal_dir = os.path.normpath(os.path.join(config.MAIN_DATA_ROOT, animal_str))
    raw_dir = os.path.join(animal_dir, config.PRETRAINING_RAW_DATA_SUBDIR)
    save_dir = os.path.join(animal_dir, config.PRETRAINING_PROCESSED_DATA_SUBDIR)
    os.makedirs(save_dir, exist_ok=True)
    behav_file_list = helper.list_files_by_time(raw_dir, file_type='.txt', print_names=0)
    for session, name in enumerate(behav_file_list):
        try:
            print(f"Processing session: {name} ...")
            session_dir = os.path.join(raw_dir, name)
            pi_events, session_identifier = process_behavior_data(session_dir)
            filename_base = f"{animal_str}_{session_identifier}_pi_events_processed"
            file_path = os.path.join(save_dir, f"{filename_base}.parquet")
            pi_events.to_parquet(file_path)
            print(f"Successfully processed and saved: {file_path}")
        except Exception as e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"⚠️ ERROR: Failed to process session: {name}")
            print(f"   Error details: {e}")
            print(f"   Skipping this session and continuing to the next.")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            continue


def main():
    # animal_str = "SZ036"
    for animal_str in ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043", "RK007", "RK008"]:
        try:
            print(f"--- Start Processing Animal {animal_str} ---")
            process_animal_data_and_save(animal_str)
            print(f"--- End Processing Animal {animal_str} ---")
        except Exception as e:
            print(f"**************************************************")
            print(f"⚠️ ERROR: Failed to process animal: {animal_str}")
            print(f"   Error details: {e}")
            print(f"   Skipping this animal and continuing to the next.")
            print(f"**************************************************")

            continue

if __name__ == "__main__":
    main()