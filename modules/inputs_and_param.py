import pandas as pd
import numpy as np
from fmpy import *
import shutil
import os
import datetime
from modules.plotting_results import plot
from modules.resilience_index import calculate_resilience
from modules.oemof_model import calculate_oemof_model

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


def get_start_values(scenarios, filename):
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
    path = os.path.join("input", "common", "dimension_scenarios")

    df_params = pd.read_csv(
        os.path.join(path, filename), delimiter=";").set_index("Parameter")

    start_values_dict = df_params.to_dict()
    start_values_dict = {k: v for k, v in start_values_dict.items()
                         if k in scenarios}

    return start_values_dict


def get_simulation_period(err_file):
    """Finding the simulation period corresponding to the error"""

    fn = os.path.join("input", "modelica", "error_scenarios", err_file)

    try:
        err_df = pd.read_csv(fn, delimiter=";", index_col="sec")
    except ValueError:
        err_df = pd.read_csv(fn, delimiter=",", index_col="sec")

    simulation_year = datetime.datetime.strptime(err_df["date"].values[0], "%d.%m.%Y %H:%M").year
    if "short" in err_file or "medium" in err_file:
        duration = 3  # in days
    else:
        duration = 10  # in days
    if "winter" in err_file:
        starting_point = "09-01"
        starting_point = datetime.datetime.strptime(starting_point, "%d-%m")
    elif "summer" in err_file:
        starting_point = "17-07"
        starting_point = datetime.datetime.strptime(starting_point, "%d-%m")
    else:
        starting_point = "17-04"
        starting_point = datetime.datetime.strptime(starting_point, "%d-%m")

    starting_point = starting_point.replace(year=simulation_year)
    starting_point_sec = int((starting_point - datetime.datetime(year=simulation_year, month=1, day=1)).total_seconds())
    starting_point_index = int(starting_point_sec / 900)
    endpoint = starting_point + datetime.timedelta(days=duration)
    endpoint_sec = int((endpoint - datetime.datetime(year=simulation_year, month=1, day=1)).total_seconds())
    endpoint_index = int(endpoint_sec / 900)


    return starting_point_index, endpoint_index


def get_inputs(
        sch_profiles,
        #simulation_period,
        err_file="reference.CSV",
        T_file="T_amp_input.CSV",
        load_file="LoadProfiles_input.CSV",
):
    """
    Function to read and prepare the inputs to pass them to the FMU simulation.

    Arguments:
        err_file: CSV file name for the errors. Default: "ErrorProfiles_input.CSV"
        sch_profiles: CSV file name for the scheduled profiles. Default: "ScheduleProfiles_input.CSV"
        T_file: CSV file name for the temperature. Default: "T_amp_input.CSV"
        load_file: CSV file name for the load profiles. Default: "LoadProfiles_input.CSV"

    Returns:
        input: Input values in the form of a structured array directly formatted as the function fmpy.simulate_fmu
        requires.

    Simulation period: Instead of taking it as input, it is made dependent on the error file that is being used.
    If it is short or medium duration we simulate from a day before the error till a day after the error (3 days in
    total) and, if it is long duration, we simulate from a day before to 2 day afterwards (so 10 days in total).
    Then there are 3 periods where the error might occur:
        winter >>> 10.01
        summer >>> 18.07
        transitional >>> 18.04

    """
    fn = os.path.join("input", "modelica", "error_scenarios", err_file)
    path_common = os.path.join("input", "common")

    try:
        err_df = pd.read_csv(fn, delimiter=";", index_col="sec")
    except ValueError:
        err_df = pd.read_csv(fn, delimiter=",", index_col="sec")

    T_df = pd.read_csv(os.path.join(path_common, T_file), delimiter=";", index_col="Time")
    load_df = pd.read_csv(os.path.join(path_common, load_file), delimiter=";", index_col="HOUR")


    # Demand power and heat are calculated as the corresponding additions of all power and heat demands
    load_df["DemandPower"] = load_df["E_el_HH"] + load_df["E_el_GHD"]
    load_df["DemandHeat"] = load_df["E_th_RH_HH"] + load_df["E_th_TWE_HH"] + load_df["E_th_RH_GHD"] + \
                            load_df["E_th_TWE_GHD"] + load_df["E_th_KL_GHD"]

    load_df.index.rename('time', inplace=True)

    # defining input variables and filling in the values for every 15 min (900 sec) without interpolation
    time = load_df.index

    if sch_profiles.index:
        sch_df = sch_profiles
        sch_df = pd.DataFrame(sch_df.values.repeat(4, axis=0), columns=sch_df.columns)  # adjusting the sizes


        u_HeatPump_scheudle = sch_df["u_HeatPump_scheudle"]  #.iloc[int(starting_point_index):int(endpoint_index)]
        u_Electrolyzer_scheudle = sch_df["u_Electrolyzer_scheudle"]  #.iloc[int(starting_point_index):int(endpoint_index)]
        u_Boiler_scheudle = sch_df["u_Boiler_scheudle"]  #.iloc[int(starting_point_index):int(endpoint_index)]
        u_CHP_scheudle = sch_df["u_CHP_scheudle"]  #.iloc[int(starting_point_index):int(endpoint_index)]

    else:
        shape = time.shape[0]
        u_Electrolyzer_scheudle = np.repeat(-1, shape)
        u_Boiler_scheudle = np.repeat(-1, shape)
        u_CHP_scheudle = np.repeat(-1, shape)
        u_HeatPump_scheudle = np.repeat(-1, shape)

    err_df = pd.DataFrame(err_df.values.repeat(4, axis=0), columns=err_df.columns)
    T_df = pd.DataFrame(T_df.values.repeat(4, axis=0), columns=T_df.columns)


    T_amb = T_df["T_amp"]  #.iloc[int(starting_point_index):int(endpoint_index)]
    u_HeatPump1_error = err_df["Heatpump1"]  #.iloc[int(starting_point_index):int(endpoint_index)]
    u_HeatPump2_error = err_df["Heatpump2"]  #.iloc[int(starting_point_index):int(endpoint_index)]
    u_Electrolyzer_error = err_df["Electrolysis"]  #.iloc[int(starting_point_index):int(endpoint_index)]
    u_Boiler_error = err_df["Boiler"]  #.iloc[int(starting_point_index):int(endpoint_index)]
    u_CHP_error = err_df["CHP"]  #.iloc[int(starting_point_index):int(endpoint_index)]
    u_loadProfile_DemandPower_kW = load_df["DemandPower"]  #.iloc[int(starting_point_index):int(endpoint_index)]
    u_loadProfile_DemandHeat_kW = load_df["DemandHeat"]  #.iloc[int(starting_point_index):int(endpoint_index)]

    # the missing inputs are filled up with zeros
    el_costs_extern = np.zeros(time.shape[0])    # np.repeat(0.0, np.array(load_df["DemandPower"]).shape)
    co2_extern = np.zeros(time.shape[0])
    u_loadProfile_DemandEMob_kW = np.zeros(time.shape[0])
    u_loadProfile_ProductionPV_kW = np.zeros(time.shape[0])


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


def get_profiles():
    pass


def simulation(
        error_files,
        dimension_scenarios,
        dimension_filename="Parameter_Values.csv",
        fmu_filename="FMU_Container.fmu",
        make_plot=True,
        store_results=None,
        simulation_period=("01-01", 14),
        schedule_profiles_filenames=None
):
    """

    Parameters
    ----------
    schedule_profiles
    fmu_filename : str
        Filename of fmu file.
    error_files : list
        Filenames of error scenarios.
    dimension_scenarios : list
        Filenames of dimension scenarios.
    make_plot : bool
        Shows some plots.
    store_results : str
        Filename for storing the results. If none, nothing ist stored.
    simulation_period : tuple
        Date of start (DD-MM), number of days

    Returns
    -------

    """
    start_values_dict = get_start_values(
        filename=dimension_filename,
        scenarios=dimension_scenarios,
    )

    # extract the FMU to a temporary directory
    unzipdir = extract(os.path.join("input", "modelica", "fmu", fmu_filename))

    # read the model description
    model_description = read_model_description(unzipdir)

    # instantiate the FMU
    fmu_instance = instantiate_fmu(unzipdir, model_description, 'CoSimulation')

    for scenario in dimension_scenarios:

        # Finding the right fahrplan for the current scenario
        if schedule_profiles_filenames:
            for file in schedule_profiles_filenames:
                if scenario in file:
                    print(scenario)
                    schedule_profiles_filename = file
                    break

            schedule_profiles = pd.read_csv(
                os.path.join("input", "modelica", "profiles",
                             schedule_profiles_filename)
            )
        else:
            schedule_profiles = None
        # for each error file, the structured array for the inputs is generated
        # and each scenario is simulated, the
        # corresponding results saved in a csv file
        for error_file in error_files:

            start_time, end_time = get_simulation_period(error_file)

            print(start_time, end_time)

            inputs = get_inputs(
                err_file=error_file,
                sch_profiles=schedule_profiles,
                #simulation_period=simulation_period
            )
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
                                  debug_logging=True,
                                  start_time=start_time,
                                  stop_time=end_time)

            print("FMU simulation finished.")

            df_res = pd.DataFrame(result).set_index("time")


            csv_filename = "results_" + str(scenario) + "_" + error_file
            df_res.to_csv(os.path.join(store_results, "data", csv_filename))

            print("ACTUAL TIME (after simulation)", df_res.index)

            if make_plot:
                plot(data_file=csv_filename, store_results=store_results, scenarios=dimension_scenarios)

    # free the FMU instance and unload the shared library
    fmu_instance.freeInstance()

    # delete the temporary directory
    shutil.rmtree(unzipdir, ignore_errors=True)
