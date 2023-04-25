import pandas as pd
import numpy as np
import os
from modules.plotting_results import resilience_box_plot
import parse


filenames = ["results/data/results_Scenario A_ErrorProfiles_input.csv",
             "results/data/results_Scenario A_ErrorProfiles_input_Boiler_14_10_18.csv",
             "results/data/results_Scenario A_ErrorProfiles_input_CHP_13_1_18.csv",
             "results/data/results_Scenario B_ErrorProfiles_input.csv",
             "results/data/results_Scenario B_ErrorProfiles_input_Boiler_14_10_18.csv",
             "results/data/results_Scenario B_ErrorProfiles_input_CHP_13_1_18.csv",
             "results/data/results_Scenario C_ErrorProfiles_input.csv",
             "results/data/results_Scenario C_ErrorProfiles_input_Boiler_14_10_18.csv",
             "results/data/results_Scenario C_ErrorProfiles_input_CHP_13_1_18.csv"]

Tband = 5  # this is +/- 5 as a tolerance band
dTnorm = 5  # [C] # same as the Tband?
dtNORM = 3600  # [s]


def maximum_deviation(df, dTnorm=dTnorm):
    """Takes the Temperature data frame of a scenario and it normalizes the maximum deviation dividing by
    a predefined constant, dTnorm.

    Arguments:
        df: DataFrame with the info related to the Temperature evolution in the system in a certain scenario.
        dTnorm: float. Normalization constant for the maximum deviation in the temperature (with respect to the set
        temperature. Default: 5.

    Returns:
        MD: float. (Normalized) maximum deviation.
        """

    tmax = df['dx'].max()
    MD = tmax / dTnorm

    return MD


def recovery_time(df, dtnorm=dtNORM):
    """
    It calculates the time that the system needs to go back to the set temperature after a large change in the
    temperature.

    Arguments:
        df: DataFrame with the info related to the Temperature evolution in the system in a certain scenario.
        dtnorm: float. Normalization constant for the maximum deviation in the temperature (with respect to the set
        temperature. Default: 5.

    Returns:
        RT: float. Recovery time of the system.
    """
    if df.loc[df["dx"] != 0].empty:
        tout = 0
        tin = df["dx"].index[-1]
    else:
        tout = df.loc[df["dx"] != 0].index[0]
        tin = df.loc[df["dx"] != 0].index[-1]

    RT = (tin - tout) / dtnorm

    return RT


def performance_loss(df, dtnorm=dtNORM):
    """
       It calculates the performance loss of the system as the ratio between the sum of all temperature deviations
       for each time step and the product of the temperature and time normalization constants.

       Arguments:
           df: DataFrame with the info related to the Temperature evolution in the system in a certain scenario.
           dtnorm: float. Time normalization constant. Default: 3600.

       Returns:
           PL: float. Performance loss of the system.
       """
    area = df['dx'].sum()
    area_norm = dTnorm * dtnorm

    PL = area / area_norm

    return PL


def resilience_index(MD, RT, PL):
    """Calculation of the resilience index of the system as the inverse of the product of the maximum deviation (MD),
    recovery time (RT) and performance loss (PL) plus 1.

    Arguments:
        MD: float. Maximum deviation.
        RT: float. Recovery time.
        PL: float. Performance loss.

    Returns:
        RI: float. Resilience index.
        """
    IRI = MD * RT * PL
    RI = 1 / (1 + IRI)

    return RI


def prepare_dataframe(filename):
    """Reads the DataFrame with the results of the simulation for a certain scenario and transforms it
    to contain exlusively the information related to the temperature of the system, adding columns to
    analyze the deviation of this with respect to the set temperature of the system.

    Arguments:
         filename: string. Name of the data file with the simulation results.

    Returns:
        df: DataFrame with the temperature information.
        """
    df = pd.read_csv(filename)

    # Converting the temperature to Celsius
    df["fMU_PhyModel.temperature_HeatGrid_FF.T"] = df["fMU_PhyModel.temperature_HeatGrid_FF.T"] - 273.15

    # Selecting the temperature info and simplifying the column names
    temps = df.loc[:, ["fMU_PhyModel.temperature_HeatGrid_FF.T", "controller.u_T_HeatGrid_FF_set"]]
    temps.rename(columns={'fMU_PhyModel.temperature_HeatGrid_FF.T': 'Temperature',
                          'controller.u_T_HeatGrid_FF_set': "Set Temperature"}, inplace=True)

    df = temps

    # new data columns are added to define the temperature band across the simulation time
    df["dT1"] = df["Set Temperature"] + Tband
    df["dT2"] = df["Set Temperature"] - Tband

    # Adding the time column and setting it as index
    time = np.linspace(0, 900 * len(df["Temperature"]), len(df["Temperature"]), endpoint=False)
    df["Time"] = time
    df = df.set_index("Time")

    # Adding new columns that take the value of the deviation up or down of the T band or 0 if there is
    # no deviation of the temperature outside the T band for that instant and that direction
    df["dx1"] = np.where((df['Temperature'] >= df["dT1"]), df['Temperature'] - df["dT1"], 0)
    df["dx2"] = np.where((df['Temperature'] <= df["dT2"]), df["dT2"] - df["Temperature"], 0)

    # Lastly, a column with the total deviation for each time step
    df["dx"] = df["dx1"] + df["dx2"]

    return df


def calculate_resilience(store_results, make_boxplot=True, scenarios=["A", "B", "C"],
                         errors=["Boiler_14_10_18", "CHP_13_1_18"]):
    """
    Function to calculate the resilience index corresponding to each different simulation (scenario and error)
    and creating a table (and plot) of the resulting values, along with the magnitudes required to obtain it.

    Arguments:
        make_boxplot: boolean. Whether the plot should be made or not. Default: True.
        scenarios: list of strings. Scenarios names. Default: ["A", "B", "C"].
        errors: list of strings. Error names. Default:["Boiler_14_10_18", "CHP_13_1_18"].
        store_results: string. Path where the CSV table and the plot should be saved.

    """

    # The data folder is searched for all CSV results files
    files_list = os.listdir(os.path.join(store_results, "data"))
    csv_list = []
    for file in files_list:
        if ".CSV" in file or ".csv" in file:
            csv_list.append(file)

    resilience_info = {}  # dictionary where all the resilience info for the table will be saved

    for file in csv_list:
        title = ""
        for scenario in scenarios:
            if scenario in file:
                title = title + scenario
        for error in errors:
            if error in file:
                title = title + " with error " + error.replace("_", " ")

        # for each file, the corresponding DataFrame with the temperature info is created
        df = prepare_dataframe(os.path.join(store_results, "data", file))

        # the time normalization constant is calculated
        t_dis_end = df.loc[df["dx"] == df["dx"].max()].index.values[0]  # time index of max deviation point
        t_dis_start = df["dx"].ne(0).idxmax()  # first non-zero element´s time index
        dtnorm = t_dis_end - t_dis_start

        # If no point is found outside the T band, then the normalization time interval takes the default value
        # of an hour, 3600
        if dtnorm == 0:
            dtnorm = dtNORM

        # All relevant magnitudes for the resilience index are calculated along with it and everything is saved
        MD = maximum_deviation(df)
        RT = recovery_time(df, dtnorm=dtnorm)
        PL = performance_loss(df, dtnorm=dtnorm)
        RI = resilience_index(MD, RT, PL)

        resilience_info.update(({title: [MD, RT, PL, RI]}))


    index = ["MD", "RT", "PL", "RI"]
    resilience = pd.DataFrame(resilience_info, index=index)
    # resilience["Average"] = resilience.mean(axis=1)

    csv_filename = "resilience.csv"
    resilience.to_csv(os.path.join(store_results, "data", csv_filename))

    if make_boxplot:
        resilience_box_plot(csv_filename, scenarios=scenarios, errors=errors, store_results=store_results)

