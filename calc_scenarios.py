import datetime
import os

from modules.inputs_and_param import simulation
from modules.plotting_results import temperature_control
from modules.resilience_index import calculate_resilience
from modules.scenarios_and_errors import read_scenarios_names, generating_error_files_list
from modules.res_tools_flexible import anlagen_table_convertor, resilience_attributes_calculation

#scenarios = [
#    "Scenario A",
#    "Scenario B",
#    "Scenario C",
#]

#error_files = [
    # "ErrorProfiles_input.CSV",
    # "ErrorProfiles_input_Boiler_14_10_18.CSV",
    # "ErrorProfiles_input_CHP_13_1_18.CSV",
    # "ErrorProfiles_input_no-errors.CSV",
# ]

# Input files
# err_file = "ErrorProfiles_input.CSV"

sch_files = os.listdir(os.path.join("input", "modelica", "profiles"))#"2030-syn-gas-low_Scenario A_ER-0.5_01-01-2018_365.CSV"  # "ScheduleProfiles_input.CSV"
T_file = "T_amp_input.CSV"
load_file = "LoadProfiles_input.CSV"

if __name__ == '__main__':
    print(sch_files)

    execution_time = datetime.datetime.now()
    ex_time = execution_time.strftime("%m%d%Y_%H_%M_%S")
    store_results = os.path.join("results", ex_time)
    os.mkdir(store_results)
    os.mkdir(os.path.join(store_results, "data"))
    os.mkdir(os.path.join(store_results, "plots"))

    scenarios = read_scenarios_names("Parameter_Values.csv")
    print(scenarios)
    error_files, error_names = generating_error_files_list()

    simulation(
        dimension_filename="Parameter_Values.csv",
        dimension_scenarios=scenarios[:2],
        error_files=error_files[:2],
        make_plot=True,
        store_results=store_results,
        #simulation_period=("01-01", 14),
        fmu_filename="FMU_Container.fmu",
        schedule_profiles_filenames=None  #,
    )

    print("Simulation done.")


    # Data treatment
    calculate_resilience(make_boxplot=True, store_results=store_results, scenarios=scenarios, errors=error_names)
    anlagen_table_convertor(store_results=store_results, scenarios=scenarios)
    resilience_attributes_calculation(store_results=store_results)

    # Plots
    temperature_control(store_results=store_results, scenarios=scenarios, errors=error_names)

