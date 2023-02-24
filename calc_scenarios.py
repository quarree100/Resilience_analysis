from modules.inputs_and_param import simulation
from modules.plotting_results import temperature_control
from modules.resilience_index import calculate_resilience
from modules.plotting_results import anlagen_table_convertor
from modules.plotting_results import radar_chart
import datetime
import os

scenarios = [
    "Scenario A",
    "Scenario B",
    "Scenario C",
]

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

if __name__ == '__main__':

    execution_time = datetime.datetime.now()
    ex_time = execution_time.strftime("%m%d%Y_%H_%M_%S")
    store_results = os.path.join("results", ex_time)
    os.mkdir(store_results)
    os.mkdir(os.path.join(store_results, "data"))
    os.mkdir(os.path.join(store_results, "plots"))

    simulation(
        dimension_filename="Parameter_Values.csv",
        dimension_scenarios=scenarios,
        error_files=error_files,
        make_plot=True,
        store_results=store_results,
        simulation_period=("01-01", 14),
        fmu_filename="FMU_Container.fmu",
        schedule_profiles_filename="ScheduleProfiles_input.CSV"  #,
    )

    print("Simulation done.")

    temperature_control(store_results=store_results)
    calculate_resilience(make_boxplot=True, store_results=store_results)

    anlagen_table_convertor()

    radar_chart(store_results=store_results)

