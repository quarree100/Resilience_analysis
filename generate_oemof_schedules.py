import os
import modules.oemof_model as om

# Input

dimension_scenario = "Scenario-A"
simulation_period = ("01-01-2022", 10)
factor_emission_reduction = 0.5
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

schedules.to_csv(os.path.join("results", "oemof", scenario_name + ".csv"))

schedules_for_modelica = om.prepare_schedules(schedules)

schedules.to_csv("")
