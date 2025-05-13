import numpy as np
import pandas as pd
import os
#import datetime


def read_scenarios_names(input_file="Parameter_Values.csv"):
    """
    Reads the scenarios names from the Parameter_Values.csv input file and gives them back as a list.

    Arguments:
        input_file: string. Name of the csv file with the scenarios names in the columns. Default: Parameter_Values.csv

    Returns:
        scenarios: list of strings. Scenarios names.

    """
    input_path = os.path.join("input", "common", "dimension_scenarios", input_file)

    df = pd.read_csv(input_path, delimiter=";")

    scenarios = []
    for scenario in df.columns[3:]:
        scenarios.append(scenario)

    return scenarios

def generating_error_files(input_file="Error Scenarios.xlsx"):
    """
    Takes an excel file as reference and generates the different error files from the information contained on
    the table: the name of the file, the element of the system that is not working and the time interval in which
    said element is out.

    Args:
        input_file: string. Name of the excel file with all the error information. Default: Error Scenarios.xlsx.

    """

    path = os.path.join("input", "modelica", "error_scenarios")

    df = pd.read_excel(os.path.join(path, input_file), engine="openpyxl")

    new_columns = df.columns[8:]
    for index in df.index[1:]:
        filename = df.iloc[index]["error type"] + ".csv"
        new_file_index = np.linspace(0, 31532400, 8760, dtype=int)
        new_df = pd.DataFrame(data=1, index=new_file_index, columns=new_columns)
        new_df["date"] = pd.date_range("2018-01-01", periods=new_df.shape[0], freq="H").strftime("%d.%m.%Y %H:%M")

        new_df.index.name = "sec"

        start_time = df.iloc[index].start
        end_time = df.iloc[index].end

        num = int((end_time - start_time) / 3600)

        filtered_indices_list = list(np.linspace(start_time, end_time, num, endpoint=False, dtype=int))


        for filtered_index in filtered_indices_list:
            new_df.loc[new_df.index == filtered_index, new_columns] = df.iloc[index][new_columns].array

        new_df.to_csv(os.path.join(path, filename))

def generating_error_files_list():
    """
    Generates a list of all error file names with and without ".csv"

    Returns:
        error_files_list: list of strings. List of the names of all error files.
        error_names: list of strings. List of the names of the error files without the ".csv" termination.

    """

    files_list = os.listdir(path=os.path.join("input", "modelica", "error_scenarios"))

    error_files_list = []
    error_names = []

    for file in files_list:
        if ".csv" in file:
            error_files_list.append(file)
            error_names.append(file.strip(".csv"))
        if ".CSV" in file:
            error_files_list.append(file)
            error_names.append(file.strip(".CSV"))

    return error_files_list, error_names