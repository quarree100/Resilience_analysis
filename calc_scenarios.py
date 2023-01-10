from modules.inputs_and_param import simulation
from modules.plotting_results import temperature_control
from modules.resilience_index import calculate_resilience

scenarios = [
    "Scenario A",
    # "Scenario B",
    # "Scenario C",
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

    simulation(
        dimension_filename="Parameter_Values.csv",
        dimension_scenarios=scenarios,
        error_files=error_files,
        make_plot=True,
        store_results="results/data/",
        simulation_period=("01-01", 14),
        fmu_filename="FMU_Container.fmu",
        schedule_profiles_filename="ScheduleProfiles_input.CSV"  #,
    )

    print("Simulation done.")

    temperature_control()
    calculate_resilience(make_boxplot=True)
