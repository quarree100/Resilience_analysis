import pandas as pd
import numpy as np
from fmpy import *
import shutil
import os
from plotting_results import plot
from resilience_index import calculate_resilience

fmu_filename = "FMU_Container.fmu"

parameters_filename = "Parameter_Values.csv"
scenarios = [
    "Scenario A",
    # "Scenario B",
    # "Scenario C",
]

# Outputs
outputs = ['controller.calc_Qdot_production.u_Qdot_Boiler',
           'controller.calc_Qdot_production.u_Qdot_CHP',
           'controller.calc_Qdot_production.u_Qdot_Electrolyzer',
           'controller.calc_Qdot_production.u_Qdot_Heatpump1',
           'controller.calc_Qdot_production.u_Qdot_Heatpump2',
           'controller.calc_Qdot_production.y_Qdot',
           'dynamic_Heatload_Scale.Qdot_heatload_scaled',
           'dynamic_Heatload_Scale.Qdot_heatload',
           'fMU_PhyModel.temperature_HeatGrid_FF.T',
           "controller.u_T_HeatGrid_FF_set"]

error_files = [
    # "ErrorProfiles_input.CSV",
    # "ErrorProfiles_input_Boiler_14_10_18.CSV",
    # "ErrorProfiles_input_CHP_13_1_18.CSV",
    "ErrorProfiles_input_no-errors.CSV",
]

# Input files
# err_file = "ErrorProfiles_input.CSV"
sch_file = "ScheduleProfiles_input.CSV"
T_file = "T_amp_input.CSV"
load_file = "LoadProfiles_input.CSV"


def get_start_values(filename=parameters_filename,
                     scenarios=["Scenario A", "Scenario B", "Scenario C"]):
    """
    Function to read and prepare the tech_param to pass them to the FMU simulation.

    Arguments:
        filename: Name of the CSV file with the parameter values (string)
        scenarios: List of the possible dimension_scenarios that appear in the parameter file (list of strings). Default:
         ["Scenario A", "Scenario B", "Scenario C"]

    Returns:
        start_values_dict: Dictionary of dictionaries, one per possible scenario. Each scenario dictionary contains
        all the parameter values corresponding to that scenario directly with the format that fmpy.simulate_fmu
        requires.

"""
    df_params = pd.read_csv(filename, delimiter=";").set_index("Parameter")

    # creation of an empty dictionary
    start_values_dict = {}

    # for each scenario, a dictionary is created, where the keys are the parameter names, and added as an entry to the
    # output dictionary
    for scenario in scenarios:
        scenario_dict = {}
        for jj, element in enumerate(df_params.index):
            scenario_dict.update({element: df_params[scenario][jj]})

        start_values_dict.update({scenario: scenario_dict})

    return start_values_dict


def get_inputs(err_file="ErrorProfiles_input.CSV",
               sch_file="ScheduleProfiles_input.CSV", T_file="T_amp_input.CSV",
               load_file="LoadProfiles_input.CSV"):
    """
    Function to read and prepare the inputs to pass them to the FMU simulation.

    Arguments:
        err_file: CSV file name for the errors. Default: "ErrorProfiles_input.CSV"
        sch_file: CSV file name for the scheduled profiles. Default: "ScheduleProfiles_input.CSV"
        T_file: CSV file name for the temperature. Default: "T_amp_input.CSV"
        load_file: CSV file name for the load profiles. Default: "LoadProfiles_input.CSV"

    Returns:
        input: Input values in the form of a structured array directly formatted as the function fmpy.simulate_fmu
        requires.

    """
    try:
        err_df = pd.read_csv(err_file, delimiter=";", index_col="sec")
    except ValueError:
        err_df = pd.read_csv(err_file, delimiter=",", index_col="sec")

    sch_df = pd.read_csv(sch_file, delimiter=";", index_col="sec")
    T_df = pd.read_csv(T_file, delimiter=";", index_col="Time")
    load_df = pd.read_csv(load_file, delimiter=";", index_col="HOUR")

    # Demand power and heat are calculated as the corresponding additions of all power and heat demands
    load_df["DemandPower"] = load_df["E_el_HH"] + load_df["E_el_GHD"]
    load_df["DemandHeat"] = load_df["E_th_RH_HH"] + load_df["E_th_TWE_HH"] + load_df["E_th_RH_GHD"] + \
                            load_df["E_th_TWE_GHD"] + load_df["E_th_KL_GHD"]

    load_df.index.rename('time', inplace=True)

    # defining input variables and filling in the values for every 15 min (900 sec) without interpolation
    time = load_df.index
    T_amb = np.repeat(T_df["T_amp"], 4)
    u_HeatPump1_error = np.repeat(err_df["Heatpump1"], 4)
    u_HeatPump2_error = np.repeat(err_df["Heatpump2"], 4)
    u_Electrolyzer_error = np.repeat(err_df["Electrolysis"], 4)
    u_Boiler_error = np.repeat(err_df["Boiler"], 4)
    u_CHP_error = np.repeat(err_df["CHP"], 4)
    u_HeatPump_scheudle = np.repeat(sch_df["Heatpump"], 4)
    u_Electrolyzer_scheudle = np.repeat(sch_df["Electrolysis"], 4)
    u_Boiler_scheudle = np.repeat(sch_df["Boiler"], 4)
    u_CHP_scheudle = np.repeat(sch_df["CHP"], 4)
    u_loadProfile_DemandPower_kW = np.repeat(load_df["DemandPower"], 4)
    u_loadProfile_DemandHeat_kW = np.repeat(load_df["DemandHeat"], 4)

    # the missing inputs are filled up with zeros
    el_costs_extern = np.repeat(0.0, np.array(load_df["DemandPower"]).shape)
    co2_extern = np.repeat(0.0, np.array(load_df["DemandPower"]).shape)
    u_loadProfile_DemandEMob_kW = np.repeat(0.0, np.array(load_df["DemandPower"]).shape)
    u_loadProfile_ProductionPV_kW = np.repeat(0.0, np.array(load_df["DemandPower"]).shape)

    # create a structured array that can be passed to simulate_fmu()
    dtype = np.dtype([('time', np.float64), ('T_amb', np.float64), ('u_HeatPump1_error', np.float64),
                      ('u_HeatPump2_error', np.float64), ('u_Electrolyzer_error', np.float64),
                      ('u_Boiler_error', np.float64), ('u_CHP_error', np.float64), ('u_HeatPump_scheudle', np.float64),
                      ('u_Electrolyzer_scheudle', np.float64), ('u_Boiler_scheudle', np.float64),
                      ('u_CHP_scheudle', np.float64), ('el_costs_extern', np.float64), ('co2_extern', np.float64),
                      ('u_loadProfile_DemandEMob_kW', np.float64), ('u_loadProfile_ProductionPV_kW', np.float64),
                      ('u_loadProfile_DemandPower_kW', np.float64), ('u_loadProfile_DemandHeat_kW', np.float64)])

    input = np.array(list(zip(time, T_amb, u_HeatPump1_error, u_HeatPump2_error, u_Electrolyzer_error, u_Boiler_error,
                              u_CHP_error, u_HeatPump_scheudle, u_Electrolyzer_scheudle, u_Boiler_scheudle,
                              u_CHP_scheudle, el_costs_extern, co2_extern, u_loadProfile_DemandEMob_kW,
                              u_loadProfile_ProductionPV_kW, u_loadProfile_DemandPower_kW,
                              u_loadProfile_DemandHeat_kW)), dtype=dtype)

    print("Input loaded.")

    return input


def simulation(
        fmu_filename="FMU_Container.fmu",
        outputs=outputs,
        error_files=error_files,
        dimension_scenarios=scenarios,
        make_plot=True,
        store_results=None,
):
    """

    Parameters
    ----------
    fmu_filename : str
        Filename of fmu file.
    outputs : ???
        ???
    error_files : list
        Filenames of error scenarios.
    dimension_scenarios : list
        Filenames of dimension scenarios.
    make_plot : bool
        Shows some plots.
    store_results : str
        Filename for storing the results. If none, nothing ist stored.

    Returns
    -------

    """
    start_values_dict = get_start_values()

    # extract the FMU to a temporary directory
    unzipdir = extract(fmu_filename)

    # read the model description
    model_description = read_model_description(unzipdir)

    # instantiate the FMU
    fmu_instance = instantiate_fmu(unzipdir, model_description, 'CoSimulation')

    # for each error file, the structured array for the inputs is generated and each scenario is simulated, the
    # corresponding results saved in a csv file
    for error_file in error_files:

        inputs = get_inputs(err_file=error_file)

        for scenario in dimension_scenarios:
            # reset the FMU instance instead of creating a new one
            fmu_instance.reset()

            start_values = start_values_dict[scenario]

            print("Start FMU simulation ...")

            result = simulate_fmu(unzipdir,
                                  input=inputs,
                                  output=outputs,
                                  start_values=start_values,
                                  model_description=model_description,
                                  fmu_instance=fmu_instance,
                                  debug_logging=True)

            print("FMU simulation finished.")

            df_res = pd.DataFrame(result).set_index("time")

            path = os.getcwd() + "/data"
            if not os.path.isdir(path):
                os.mkdir(path)
            csv_filename = "data/" + "results_" + str(scenario) + "_" + error_file
            df_res.to_csv(csv_filename)

            plot(csv_filename)

            # if make_plot:
            #     plot(data_file=csv_filename)

    # free the FMU instance and unload the shared library
    fmu_instance.freeInstance()

    # delete the temporary directory
    shutil.rmtree(unzipdir, ignore_errors=True)


if __name__ == '__main__':
    simulation()
    # calculate_resilience()
