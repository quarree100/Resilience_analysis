import os
import modules.oemof_model as om

# Input

dimension_scenario = "Scenario-A"

simulation_period = ("01-01-2022", 2)
factor_emission_reduction = 0.5 #between 0 (cost-optimized) and 1 (emission-optimized)

path_oemof = os.path.join("input", "solph")

scenario_name = "_".join(
        [dimension_scenario, "ER-" + str(factor_emission_reduction),
         simulation_period[0] + "-" + str(simulation_period[1])]
    )

# Calculation

schedules = om.calculate_oemof_model(
    dimension_scenario=dimension_scenario,
    simulation_period=simulation_period,
    factor_emission_reduction=factor_emission_reduction,
    path_oemof=path_oemof,
)

# export oemof results
schedules.to_csv(os.path.join("results", "oemof_raw", scenario_name + ".csv"))

# create schedules for modelica (function needs to be done)
# TODO : complete function
schedules_for_modelica = om.prepare_schedules(schedules)

# export schedules
schedules_for_modelica.to_csv(
    os.path.join("results", "modelica_schedules", scenario_name + ".csv")
)

print("finished.")
