import os

MAIN_DATA_ROOT = os.path.normpath('[the big root directory of all your data]')
PROJECT_ROOT = os.path.dirname(__file__) # can be used to save general outputs

RAW_DATA_SUBDIR = 'raw_data'
PROCESSED_DATA_SUBDIR = 'processed_data'
SUMMARY_PROCESSED_SUBDIR = 'summary'
FIGURE_SUBDIR = 'figures'
SUMMARY_FIGURE_SUBDIR = 'summary'
