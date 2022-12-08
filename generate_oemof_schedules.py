import modules.oemof_model as om

schedules = om.calculate_oemof_model(
    simulation_period=("01-01", 365),
    factor_emission_reduction=0.5,
)

schedules_for_modelica = om.prepare_schedules(schedules)

schedules.to_csv("")
