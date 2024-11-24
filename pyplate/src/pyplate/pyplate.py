# pyplate: A Python package for microwell plate data processing
# Author: [Your Name]
# License: MIT

import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import List, Dict, Tuple, Union, Optional
import warnings

class PlateError(Exception):
    """Custom exception for plate-related errors"""
    pass

def read_data(file: str, sheet: Optional[int] = None) -> pd.DataFrame:
    """
    Read data based on file extension.
    
    Args:
        file: Path to the input file
        sheet: Sheet number for Excel files
    
    Returns:
        DataFrame with the raw data
    
    Raises:
        PlateError: If file format is unsupported or file is empty
    """
    file_path = Path(file)
    file_ext = file_path.suffix.lower()
    full_name = file_path.name

    if file_ext == '.csv':
        try:
            raw_data = pd.read_csv(file, header=None, na_values=[''])
        except Exception as e:
            raise PlateError(f"Error reading CSV file: {str(e)}")
    elif file_ext in ['.xls', '.xlsx']:
        try:
            raw_data = pd.read_excel(file, sheet_name=sheet, header=None)
        except Exception as e:
            raise PlateError(f"Error reading Excel file: {str(e)}")
    else:
        raise PlateError(
            f"Unsupported file format. Expected: csv, xls, xlsx. Found: {file_ext or 'None'}"
        )

    if raw_data.empty:
        raise PlateError(
            f"Cannot read {full_name}: The input {'file' if file_ext == '.csv' else 'sheet'} is empty."
        )

    return raw_data

def get_plate_params(n_cols: int) -> dict:
    """
    Define parameters for each plate type based on number of columns.
    
    Args:
        n_cols: Number of columns in the plate
    
    Returns:
        Dictionary containing plate parameters
    
    Raises:
        PlateError: If number of columns doesn't match any known plate format
    """
    plate_info = {
        4: {'plate_type': 6, 'row_end': 3, 'increment': 4,
            'first_col_vec': list('AB'), 'first_row_vec': list(range(1, 4))},
        5: {'plate_type': 12, 'row_end': 4, 'increment': 5,
            'first_col_vec': list('ABC'), 'first_row_vec': list(range(1, 5))},
        7: {'plate_type': 24, 'row_end': 5, 'increment': 6,
            'first_col_vec': list('ABCD'), 'first_row_vec': list(range(1, 7))},
        9: {'plate_type': 48, 'row_end': 7, 'increment': 8,
            'first_col_vec': list('ABCDEF'), 'first_row_vec': list(range(1, 9))},
        13: {'plate_type': 96, 'row_end': 9, 'increment': 10,
             'first_col_vec': list('ABCDEFGH'), 'first_row_vec': list(range(1, 13))},
        25: {'plate_type': 384, 'row_end': 17, 'increment': 18,
             'first_col_vec': list('ABCDEFGHIJKLMNOP'), 'first_row_vec': list(range(1, 25))},
        49: {'plate_type': 1536, 'row_end': 33, 'increment': 34,
             'first_col_vec': [*list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), *[f'A{x}' for x in 'ABCDEF']],
             'first_row_vec': list(range(1, 49))}
    }

    if n_cols not in plate_info:
        raise PlateError("Invalid number of columns for plate format")

    return plate_info[n_cols]

def is_plate_empty(plate: pd.DataFrame) -> bool:
    """
    Check if a plate is empty (contains only NaN or empty strings).
    
    Args:
        plate: DataFrame containing plate data
    
    Returns:
        bool: True if plate is empty, False otherwise
    """
    if len(plate.index) > 1 and len(plate.columns) > 1:
        df_subset = plate.iloc[1:, 1:]
        return df_subset.isna().all().all() or (df_subset == '').all().all()
    return False

def validate_plate(raw_data: pd.DataFrame, well_id: str, file_name: str) -> dict:
    """
    Validate the plate format and structure.
    
    Args:
        raw_data: Input DataFrame containing plate data
        well_id: Name for the well ID column
        file_name: Name of the input file
    
    Returns:
        Dictionary containing plate parameters
    
    Raises:
        PlateError: If plate format is invalid or validation fails
    """
    count_columns = len(raw_data.columns)
    count_rows_actual = len(raw_data.index)

    # Check if number of columns is valid
    if count_columns not in [4, 5, 7, 9, 13, 25, 49]:
        raise PlateError(
            f"{file_name} is not a valid input file. "
            "Only 6, 12, 24, 48, 96, 384, and 1536-well plate formats are accepted."
        )

    # Get plate parameters
    params = get_plate_params(count_columns)
    
    # Calculate number of plates
    no_of_plates = sum(raw_data.isna().all(axis=1)) + 1
    count_rows_theoretical = (no_of_plates * params['increment']) - 1

    if count_rows_theoretical != count_rows_actual:
        raise PlateError(
            f"{file_name} is not a valid input file. "
            "Incorrect number of rows for the plate format."
        )

    # Split into individual plates
    plates = []
    for i in range(no_of_plates):
        start_idx = i * params['increment']
        end_idx = start_idx + params['row_end']
        plates.append(raw_data.iloc[start_idx:end_idx])

    # Validate plate names
    plate_names = [plate.iloc[0, 0] for plate in plates]
    
    # Check for empty plate names
    empty_plates = [i for i, name in enumerate(plate_names, 1) 
                   if pd.isna(name) or str(name).strip() == '']
    if empty_plates:
        raise PlateError(
            f"Empty plate name(s) found in {file_name} at position(s): {', '.join(map(str, empty_plates))}"
        )

    # Check for duplicate plate names
    non_empty_names = [name for name in plate_names if not (pd.isna(name) or str(name).strip() == '')]
    duplicates = [name for name in set(non_empty_names) if non_empty_names.count(name) > 1]
    if duplicates:
        raise PlateError(
            f"Duplicated plate name(s) found in {file_name}: {', '.join(duplicates)}"
        )

    # Check if all plates are empty
    if all(is_plate_empty(plate) for plate in plates):
        raise PlateError("All plates are empty")

    return {
        'no_of_plates': no_of_plates,
        'plate_type': params['plate_type'],
        'increment': params['increment'],
        'row_end': params['row_end'],
        'first_col_vec': params['first_col_vec'],
        'first_row_vec': params['first_row_vec']
    }

def tidy_plate(file: str, well_id: str = "well", sheet: Optional[int] = 0) -> pd.DataFrame:
    """
    Read and transform microwell plate data to a tidy DataFrame.
    
    This function reads a microwell plate shaped CSV or Excel file and returns
    a pandas DataFrame for downstream data analysis.
    
    Args:
        file: Path to the input file (CSV or Excel)
        well_id: Name for the well ID column
        sheet: Sheet number for Excel files (0-based)
    
    Returns:
        pandas.DataFrame: A tidy format DataFrame with well IDs and plate values
    
    Raises:
        PlateError: If file format is invalid or processing fails
    
    Example:
        >>> data = tidy_plate("example_12_well.csv")
        >>> print(data.head())
    """
    # Input validation
    if not isinstance(file, str):
        raise PlateError("File path must be a string")
    if not isinstance(well_id, str):
        raise PlateError("well_id must be a string")
    
    file_path = Path(file)
    if not file_path.exists():
        raise PlateError(f"File does not exist: {file}")

    # Read raw data
    raw_data = read_data(file, sheet)
    
    # Validate plate format
    params = validate_plate(raw_data, well_id, file_path.name)
    
    # Process plates
    plates_data = []
    current_row = 0
    
    while current_row < len(raw_data):
        if pd.notna(raw_data.iloc[current_row, 0]) and not re.match(r'^[A-Z]{1,2}$', str(raw_data.iloc[current_row, 0])):
            plate_id = str(raw_data.iloc[current_row, 0])
            plate = raw_data.iloc[current_row + 1:current_row + params['row_end']].copy()
            
            # Create well IDs
            wells = []
            values = []
            
            for row_idx, row in enumerate(params['first_col_vec']):
                for col_idx in range(len(params['first_row_vec'])):
                    wells.append(f"{row}{col_idx + 1:02d}")
                    values.append(plate.iloc[row_idx, col_idx + 1])
            
            plate_df = pd.DataFrame({
                well_id: wells,
                plate_id: values
            })
            plates_data.append(plate_df)
            
        current_row += params['increment']
    
    # Merge all plates
    if not plates_data:
        raise PlateError("No valid plate data found")
        
    result = plates_data[0]
    for plate_df in plates_data[1:]:
        result = result.merge(plate_df, on=well_id, how='outer')
    
    # Sort by well ID and remove rows where all values are NA
    result = result.sort_values(well_id)
    result = result.loc[~result.iloc[:, 1:].isna().all(axis=1)]
    
    # Convert data types where possible
    for col in result.columns[1:]:
        try:
            result[col] = pd.to_numeric(result[col])
        except (ValueError, TypeError):
            # Keep as is if conversion fails
            continue
    
    print(f"Plate type: {params['plate_type']}-well")
    return result