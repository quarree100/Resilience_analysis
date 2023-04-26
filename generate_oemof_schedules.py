import os
import modules.oemof_model as om

# Input

path_oemof = os.path.join("input", "solph")


simulation_period = ("03-03-2018", 4)  # max 365 days in 2018
factor_emission_reduction = 0.5 #between 0 (cost-optimized) and 1 (emission-optimized)


global_scenarios = [
    # "2020",
    "2030-syn-gas-low",
    # "2030-syn-gas-high",
    # "2050-syn-gas-low",
    # "2050-syn-gas-high",
]

dimension_scenarios = [
    "Scenario-A",
    # "Scenario-B",
    # "Scenario-C",
]

for global_sc in global_scenarios:
    for dim_sc in dimension_scenarios:

        scenario_name = "_".join(
                [global_sc, dim_sc, "ER-" + str(factor_emission_reduction),
                 simulation_period[0] + "_" + str(simulation_period[1])]
            )  # ER: emission reduction

        # Calculation

        schedules = om.calculate_oemof_model(
            dimension_scenario=dim_sc,
            simulation_period=simulation_period,
            factor_emission_reduction=factor_emission_reduction,
            path_oemof=path_oemof,
            global_scenario=global_sc,
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
