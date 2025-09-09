import csv
import matplotlib.pyplot as plt
import os
import glob
import re
import numpy as np
from matplotlib.ticker import FuncFormatter
import pandas as pd

def read_csv_data(file_path):
    """
    Reads data from a CSV file and returns two arrays.
    
    Parameters:
        file_path (str): Path to the CSV file.
        
    Returns:
        tuple: Two arrays containing the data from the CSV file.
    """
    # Initialize empty lists to store lambdas and counts
    lambdas = []
    counts = []

    # Open the CSV file in read mode
    with open(file_path, newline='') as csvfile:
        # Create a CSV reader object
        reader = csv.reader(csvfile)
        
        # Skip the header row
        next(reader)
        
        # Read each row in the CSV file
        for row in reader:
            # Add the value of the first column (lambdas) to the lambdas list
            lambdas.append(float(row[0]))
            
            # Add the value of the second column (counts) to the counts list
            counts.append(float(row[1]))
    
    # Return the arrays containing the data
    return lambdas, counts

def read_csv_polarization_dependence(file_path, file_name):
    # Specify the number of rows to skip
    skiprows = 8  # Number of rows to skip at the beginning
    
    # Read the CSV file into a DataFrame, skipping the initial rows
    df = pd.read_csv(file_path + file_name, delimiter=';', skiprows=skiprows)
    
    # Initialize an empty dictionary to store the data
    data_dict = {}

    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Extract the column values as an array and store in the dictionary
        data_dict[column] = df[column].values
    
    return data_dict

def convert_elapsed_time_to_seconds(elapsed_time_str):
    # Divide la cadena de tiempo en partes
    parts = elapsed_time_str.split(':')
    
    # Convierte cada parte a un número entero o flotante
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    milliseconds = float(parts[3])
    
    # Calcula el tiempo total en segundos
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    
    return total_seconds

def psi_vector(theta_deg, phi_deg):
    # Convert angles from degrees to radians
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    
    # Calculate the coefficients
    coef_H = np.exp(1j * phi) * np.cos(theta / 2)
    coef_V = np.sin(theta / 2)
    
    # Create the vector as a column matrix
    psi = np.array([[coef_H], [coef_V]])
    
    return psi

def fidelity_with_error(psi, rho, err_rho):
    
    # Conjugate transpose of psi
    psi_dagger = np.conjugate(psi.T)
    
    # Calculate the fidelity
    fid = np.matmul(np.matmul(psi_dagger, rho), psi)
    
    # Calculate the error
    fid_err = np.matmul(np.matmul(psi_dagger, err_rho), psi)
    
    return fid[0, 0].real, fid_err[0, 0].real  # Return the fidelity value and its error

# Function to extract a single number after "POL" and "QWP" from a string
def extract_numbers(string):
    pol_match = re.search(r'POL(\d+)', string)
    qwp_match = re.search(r'QWP(\d+)', string)
    if pol_match and qwp_match:
        return int(pol_match.group(1)), int(qwp_match.group(1))
    else:
        return None, None
    
def fill_lists(S_dict, S_dict_prime):
    list_S = []
    list_S_prime = []

    S_keys = list(S_dict.keys())
    S_prime_keys = list(S_dict_prime.keys())

    # Check if keys match in both dictionaries
    for key in S_keys:
        if key in S_prime_keys:
            list_S.append(np.array(S_dict[key]))
            list_S_prime.append(np.array(S_dict_prime[key]))

    return list_S, list_S_prime

def find_csv_filenames(path_to_dir, suffix="csv"):
    """Get names of all files in given directory with file-ending specified in suffix.
    
    Args: 
        path_to_dir(str): path to the folder containing the files in string form
        suffix(str): string, gives which file types to look for. Defaults to "csv"

    Returns: 
        result(str array): List of names of the files which end with the datatype given by suffix
    """
    if not path_to_dir:
        print("No path given!")
        raise Exception("No path given!")
    os.chdir(path_to_dir)
    result = glob.glob('*.{}'.format(suffix))
    if len(result) == 0:
        print("Empty directory!")
        raise Exception("Empty directory!")
    return result

def read_from_csv(path):
    data = []
    with open(path, 'r') as txtreader:
        for row in txtreader.readlines():
            if row[0] == '"':   # remove any leading and trailing quotations marks 
                row = row[1:-2] # I have no Idea how they get here in the first place
            if row != []:
                if row[0][0] != '#':
                    if len(row) >= 2:
                        data.append(row)
    return data

def extract_magnetic_field_from_filename(name_file):
    # Regular expression pattern to find the double number just before "T"
    pattern = r"(-?\d+\.\d+)T"
    
    # Search for the double number using regular expression
    match = re.search(pattern, name_file)

    if match:
        double_number = float(match.group(1))
        return double_number
    else:
        return None
    
def sort_counts_dict(counts_dict):
    """
    Sorts a dictionary of counts where keys are double numbers (labels) 
    and values are arrays of data corresponding to counts from a detector.
    
    Parameters:
        counts_dict (dict): Dictionary where keys are double numbers (labels) 
                            and values are arrays of data.
    
    Returns:
        dict: Sorted dictionary where keys are sorted double numbers (labels) 
              and values are corresponding sorted arrays of data.
    """
    sorted_counts_dict = dict(sorted(counts_dict.items(), key=lambda item: item[0]))
    return sorted_counts_dict

