import os
import modules.oemof_model as om
from modules.scenarios_and_errors import read_scenarios_names


# Input

path_oemof = os.path.join("input", "solph")


simulation_period = ("01-01-2018", 365)  # max 365 days in 2018
factor_emission_reduction = 0.5 #between 0 (cost-optimized) and 1 (emission-optimized)


global_scenarios = [
    "2020",
    "2030-syn-gas-low",
     "2030-syn-gas-high",
     "2050-syn-gas-low",
 "2050-syn-gas-high"]

input_scenarios = read_scenarios_names("Parameter_Values.csv")
#input_scenarios = input_scenarios[9:10]
#print("Scenario: ", input_scenarios)

for scenario in input_scenarios:

        scenario_name = "_".join(
                [scenario, "ER-" + str(factor_emission_reduction),   # after global_sc there was dim_sc
                 simulation_period[0] + "_" + str(simulation_period[1])]
            )  # ER: emission reduction

        # Calculation

        schedules, dim_kwargs = om.calculate_oemof_model(
            scenario=scenario,
            simulation_period=simulation_period,
            factor_emission_reduction=factor_emission_reduction,
            path_oemof=path_oemof,
            switch_to_hour_resolution=True,
        )

        # export oemof results
        schedules.to_csv(os.path.join("input", "modelica", "oemof_raw", scenario_name + ".csv"))

        # create schedules for modelica (function needs to be done)
        # TODO : complete function
        schedules_for_modelica = om.prepare_schedules(schedules, dim_kwargs)

        # export schedules
        schedules_for_modelica.to_csv(
            os.path.join("input", "modelica", "profiles", scenario_name + ".csv")
        )

print("finished.")
