import oemof.solph as solph
import yaml
import os
import math
import pandas as pd
from oemof.tools import logger
from oemof.network.graph import create_nx_graph
from oemof_visio import ESGraphRenderer
# from q100opt.plots import plot_es_graph
from matplotlib import pyplot as plt
import datetime

import logging


def create_solph_model(
        techparam,
        timeindex,
        timeseries,
        capacity_boiler=3000,
        capacity_chp_el=400,
        eta_el_chp=0.38,
        eta_th_chp=0.55,
        capacity_hp_air=1000,
        capacity_hp_ground=500,
        capacity_electrlysis_el=250,
        capacity_pv=1500,
        capacity_thermal_storage_m3=1000,
        weight_cost_emission=0,
):

    # initiate the logger (see the API docs for more information)
    logger.define_logging(
        logfile="oemof_example.log",
        screen_level=logging.INFO,
        file_level=logging.DEBUG,
    )

    logging.info("Initialize the energy system")

    energysystem = solph.EnergySystem(timeindex=timeindex)

    logging.info("Create oemof objects")

    b_gas = solph.Bus(label="gas")
    b_elec = solph.Bus(label="electricity")
    b_heat_generation = solph.Bus(label="heat_generation")
    b_heat_storage_out = solph.Bus(label="storage_out")
    b_heat_grid = solph.Bus(label="heat_grid")
    b_h2 = solph.Bus(label="h2")

    energysystem.add(b_gas, b_elec, b_heat_generation, b_heat_storage_out,
                     b_heat_grid, b_h2)

    var_costs_gas = \
        weight_cost_emission * techparam["gas_source"]["emission_factor"] + \
        (1 - weight_cost_emission) * techparam["gas_source"]["variable_costs"]

    gas_source = solph.Source(
        label="gas_grid",
        outputs={b_gas: solph.Flow(
            variable_costs=var_costs_gas,
            emission_factor=techparam["gas_source"]["emission_factor"],
        )}
    )

    var_costs_elec = \
        weight_cost_emission * techparam["electricity_source"][
            "emission_factor"] + \
        (1 - weight_cost_emission) * techparam["electricity_source"][
            "variable_costs"]

    elec_source = solph.Source(
        label="electricity_grid",
        outputs={b_elec: solph.Flow(
            variable_costs=var_costs_elec,
            emission_factor=techparam["electricity_source"]["emission_factor"],
        )}
    )

    pv_source = solph.Source(
        label="pv",
        outputs={b_elec: solph.Flow(
            nominal_value=capacity_pv,
            fix=timeseries["pv_normed_per_kWp"],
        )}
    )

    heat_demand = solph.Sink(
        label="demand",
        inputs={b_heat_grid: solph.Flow(
            nominal_value=1,
            fix=timeseries["Heat_demand_after_storage_kW"],
        )}
    )

    var_costs_elec_sell = \
        weight_cost_emission * techparam["electricity_sell"][
            "emission_factor"] + \
        (1 - weight_cost_emission) * techparam["electricity_sell"][
            "variable_costs"]

    elec_sell = solph.Sink(
        label="elec_sell",
        inputs={b_elec: solph.Flow(
            variable_costs=var_costs_elec_sell,
            emission_factor=techparam["electricity_sell"]["emission_factor"],
        )}
    )

    var_costs_h2_sell = \
        weight_cost_emission * techparam["hydrogen_sell"]["variable_costs"] + \
        (1 - weight_cost_emission) * techparam["hydrogen_sell"][
            "variable_costs"]

    h2_sell = solph.Sink(
        label="h2_sell",
        inputs={b_h2: solph.Flow(
            variable_costs=var_costs_h2_sell,
            emission_factor=techparam["hydrogen_sell"]["emission_factor"],
        )}
    )

    energysystem.add(h2_sell, elec_sell, heat_demand, pv_source, elec_source,
                     gas_source)

    boiler = solph.Transformer(
        label="gas_boiler",
        inputs={b_gas: solph.Flow()},
        outputs={b_heat_generation: solph.Flow(
            nominal_value=capacity_boiler,
            min=techparam["gas_boiler"]["minimum_load"],
            nonconvex=solph.options.NonConvex()
        )},
        conversion_factors={
            b_heat_generation: techparam["gas_boiler"]["efficiency"],
        }
    )

    chp = solph.Transformer(
        label="chp",
        inputs={b_gas: solph.Flow()},
        outputs={
            b_elec: solph.Flow(
                nominal_value=capacity_chp_el,
                min=techparam["chp"]["minimum_load"],
                nonconvex=solph.options.NonConvex()),
            b_heat_generation: solph.Flow()
        },
        conversion_factors={
            b_elec: eta_el_chp,
            b_heat_generation: eta_th_chp,
        }
    )

    ely = solph.Transformer(
        label="electrolysis",
        inputs={b_elec: solph.Flow(
            nominal_value=capacity_electrlysis_el,
            min=techparam["electrolysis"]["minimum_load"],
            nonconvex=solph.options.NonConvex(),
            summed_min=techparam["electrolysis"]["fullloadhours_min"],
            summed_max=techparam["electrolysis"]["fullloadhours_max"],
        )},
        outputs={
            b_h2: solph.Flow(),
            b_heat_generation: solph.Flow(),
        },
        conversion_factors={
            b_h2: techparam["electrolysis"]["efficiency_h2"],
            b_heat_generation:
                techparam["electrolysis"]["efficiency_heat_excess"],
        }
    )

    hp_air = solph.Transformer(
        label="heatpump_air",
        inputs={
            b_elec: solph.Flow()
        },
        outputs={
            b_heat_generation: solph.Flow(
                nominal_value=capacity_hp_air,
                min=techparam["heatpump_air"]["minimum_load"],
                nonconvex=solph.options.NonConvex(),
                max=timeseries["Maximum-Power_Heatpump_air"],
            ),
        },
        conversion_factors={
            b_heat_generation: timeseries["COP_Heatpump_air"],
        }
    )

    hp_ground = solph.Transformer(
        label="heatpump_ground",
        inputs={
            b_elec: solph.Flow(
                nominal_value=capacity_hp_ground,
                min=techparam["heatpump_ground"]["minimum_load"],
                nonconvex=solph.options.NonConvex(),
            )
        },
        outputs={
            b_heat_generation: solph.Flow(),
        },
        conversion_factors={
            b_heat_generation: techparam["heatpump_ground"]["cop"],
        }
    )

    energysystem.add(hp_air, hp_ground, chp, boiler, ely)

    # todo : replace the capacity calculation by a proper calculation
    storage_capa = \
        capacity_thermal_storage_m3 * \
        techparam["thermal_storage"]["capacity_per_volume"]

    thermal_storage = solph.GenericStorage(
        label="thermal_storage",
        inputs={b_heat_generation: solph.Flow()},
        outputs={b_heat_storage_out: solph.Flow()},
        nominal_storage_capacity=storage_capa,
        loss_rate=timeseries["TES_SOC_loss_factor"],
        fixed_losses_relative=timeseries["TES_fix_loss_factor"],
    )

    grid_pump = solph.Transformer(
        label="dhs_grid_pump",
        inputs={
            b_heat_storage_out: solph.Flow(),
            b_elec: solph.Flow(),
        },
        outputs={
            b_heat_grid: solph.Flow(),
        },
        conversion_factors={
            b_elec: 0.015,
            b_heat_storage_out: 1,
            b_heat_grid: 1,
        }
    )

    energysystem.add(grid_pump, thermal_storage)

    # plot_es_graph(energysystem, show=True)

    # gr = ESGraphRenderer(energy_system=energysystem, filepath="energy_system",
    #                      img_format="png")
    # gr.view()

    return energysystem


def solve_model(energysystem, emission_limit=1000000000):
    solver = "gurobi"  # 'glpk', 'gurobi',....
    debug = False  # Set number_of_timesteps to 3 to get a readable lp-file.
    solver_verbose = True  # show/hide solver output

    logging.info("Optimise the energy system")

    # initialise the operational model
    model = solph.Model(energysystem)

    solph.constraints.generic_integral_limit(
        model, keyword='emission_factor', limit=emission_limit
    )

    # This is for debugging only. It is not(!) necessary to solve the problem and
    # should be set to False to save time and disc space in normal use. For
    # debugging the timesteps should be set to 3, to increase the readability of
    # the lp-file.
    if debug:
        filename = os.path.join(
            solph.helpers.extend_basic_path("lp_files"), "basic_example.lp"
        )
        logging.info("Store lp-file in {0}.".format(filename))
        model.write(filename, io_options={"symbolic_solver_labels": True})

    # if tee_switch is true solver messages will be displayed
    logging.info("Solve the optimization problem")

    solver_cmdline_options = {
        # 'threads': 1,
        # gurobi
        'MIPGap': 0.001,
    }

    model.solve(solver=solver, solve_kwargs={"tee": solver_verbose},
                cmdline_options=solver_cmdline_options
                )

    # The processing module of the outputlib can be used to extract the results
    # from the model transfer them into a homogeneous structured dictionary.

    # add results to the energy system to make it possible to store them.
    energysystem.results["main"] = solph.processing.results(model)
    energysystem.results["meta"] = solph.processing.meta_results(model)

    energysystem.results["meta"]["emission_value"] = \
        model.integral_limit_emission_factor()

    return energysystem


# print and plot some results
def plot_results(esys):
    """

    Args:
        esys:

    Returns:

    """

    results = esys.results["main"]

    heat_gen = solph.views.node(results, "heat_generation")
    heat_store = solph.views.node(results, "thermal_storage")
    elec = solph.views.node(results, "electricity")

    print(heat_gen["sequences"].sum())
    print(heat_store["sequences"].sum())
    print(elec["sequences"].sum())

    fig1, ax = plt.subplots(figsize=(10, 5))
    heat_gen["sequences"].plot(ax=ax)
    plt.show()

    fig2, ax = plt.subplots(figsize=(10, 5))
    heat_store["sequences"].plot(ax=ax)
    plt.show()

    fig3, ax = plt.subplots(figsize=(10, 5))
    elec["sequences"].plot(ax=ax)
    plt.show()


def calculate_oemof_model(
        dimension_scenario,
        simulation_period=("01-01-2022", 365),
        factor_emission_reduction=0.5,
        path_oemof=os.path.join("input", "solph"),
        path_common=os.path.join("input", "common"),
        show_plots=True,
):
    """
    Calculates the oemof-solph model and return the unit commitment
    schedule for the heat generation units.

    For the technical data the file `parameter.yaml` is used in the
    oemof input data folder.

    As timeseries, the `Timeseries_15min.csv` of the oemof input folder
    is used.

    Parameters
    ----------
    simulation_period : tuple
        Start date and length of period in days
    factor_emission_reduction : scalar
        Factor that describes the relative emission reduction.
        0 : cost optimal case
        1 : emission optimal case
    path_oemof
        Path to the oemof input data folder
    path_common
        Path to the common input data folder
    Returns
    -------

    """

    dim_sc = pd.read_csv(os.path.join(
        path_common, "dimension_scenarios", dimension_scenario + ".csv"),
        index_col=0,
    )

    volumen_tes = \
        (0.5 * dim_sc.loc["d_tes", "Value"]) ** 2 * math.pi * \
        dim_sc.loc["h_tes", "Value"]

    cap_hp_air = dim_sc.loc["ScaleFactor_HP1", "Value"] * 500 +\
                 dim_sc.loc["ScaleFactor_HP2", "Value"] * 500

    cap_hp_ground = 0

    dim_kwargs = {
        "capacity_boiler": dim_sc.loc["capQ_th_boiler", "Value"],
        "capacity_chp_el": dim_sc.loc["capP_el_chp", "Value"],
        "capacity_hp_air": cap_hp_air,
        "capacity_hp_ground": cap_hp_ground,
        "capacity_electrlysis_el": dim_sc.loc["capP_el_electrolyser", "Value"],
        "capacity_pv": dim_sc.loc["capP_el_pv", "Value"],
        "capacity_thermal_storage_m3": volumen_tes,
        "eta_el_chp": dim_sc.loc["eta_el_chp", "Value"],
        "eta_th_chp": dim_sc.loc["eta_th_chp", "Value"],
    }

    tech_param = os.path.join(path_oemof, "parameter.yaml")

    with open(tech_param) as file:
        tech_param = yaml.safe_load(file)

    timeseries = pd.read_csv(os.path.join(path_oemof, "Timeseries_15min.csv"),
                             sep=",", skiprows=[0])

    timeseries.index = pd.DatetimeIndex(
        pd.date_range(start="01-01-2022", freq="15min", periods=8760 * 4)
    )

    # Create and solve oemof-solph model

    start = pd.to_datetime(simulation_period[0], yearfirst=False)
    end = start + pd.Timedelta(simulation_period[1], unit="D")
    time_slice = timeseries[start:end]

    # what is the minimum possible emission value?

    logging.info("Calculate minimum emission limit")

    esys_emission = create_solph_model(
        techparam=tech_param,
        timeindex=time_slice.index,
        timeseries=time_slice,
        weight_cost_emission=1,
        **dim_kwargs
    )
    esys_emission = solve_model(esys_emission)
    emission_min = esys_emission.results["meta"][
        "objective"]  # emission in [kg]

    # what is the emission value in the cost optimal case?

    logging.info("Calculate maximum emission limit")

    esys_cost = create_solph_model(
        techparam=tech_param,
        timeindex=time_slice.index,
        timeseries=time_slice,
        weight_cost_emission=0,
        **dim_kwargs
    )

    esys_max = solve_model(esys_cost)

    emission_max = esys_max.results["meta"]["emission_value"]
    cost_max = esys_max.results["meta"]["objective"]

    esys_min = solve_model(esys_cost, emission_limit=emission_min + 0.1)
    costs_min = esys_min.results["meta"]["objective"]

    emission_limit_mid = \
        factor_emission_reduction * (
                emission_max - emission_min) + emission_min

    logging.info("Calculate scenario with `factor_emission_reduction`")

    esys_mid = solve_model(esys_cost, emission_limit=emission_limit_mid)

    emission_mid = esys_mid.results["meta"]["emission_value"]
    costs_mid = esys_mid.results["meta"]["objective"]

    # plots
    if show_plots:
        fig, ax = plt.subplots()
        ax.scatter(emission_max, cost_max, color='r')
        ax.scatter(emission_min, costs_min, color='b')
        ax.scatter(emission_mid, costs_mid, color='tab:orange')
        ax.set_xlabel('Emission [kg]')
        ax.set_ylabel('Costs [€]')
        ax.grid()
        # ax.set_title('scatter plot')
        plt.show()

    # get results
    results = esys_mid.results["main"]

    th_storage = solph.views.node(results, "thermal_storage")["sequences"]
    boiler = solph.views.node(results, "gas_boiler")["sequences"]
    ely = solph.views.node(results, "electrolysis")["sequences"]
    chp = solph.views.node(results, "chp")["sequences"]
    hp_air = solph.views.node(results, "heatpump_air")["sequences"]
    hp_ground = solph.views.node(results, "heatpump_ground")["sequences"]

    list_comps = [boiler, chp, hp_air, hp_ground, ely, th_storage]

    df_restuls = pd.concat(list_comps, axis=1)

    if show_plots:
        for comp in list_comps:
            comp.plot()
            plt.show()

    return df_restuls


def prepare_schedules(df):
    """
    This function should prepare the oemof schedules for the modelica input.

    Parameters
    ----------
    df : pandas.DateFrame
        Table with the results of the oemof-solph optimization.

    Returns
    -------
    pandas.DateFrame : With the timeseries format for the modelica input.
    """

    df_modelica = pd.DataFrame()

    return df_modelica


if __name__ == '__main__':

    # perspective function arguments

    simulation_period = ("15-02-2022", 14)  # start date, length of period in days

    # more or less fixed input paths (no function attributes)

    path_oemof = os.path.join("..", "input", "solph")
    path_common = os.path.join("..", "input", "common")

    tech_param = os.path.join(path_oemof, "parameter.yaml")

    timeseries = pd.read_csv(os.path.join(path_oemof, "Timeseries_15min.csv"),
                             sep=",")

    timeseries.index = pd.DatetimeIndex(
        pd.date_range(start="01-01-2022", freq="15min", periods=8760*4)
    )

    with open(tech_param) as file:
        tech_param = yaml.safe_load(file)

    # Create and solve oemof-solph model

    start = pd.to_datetime(simulation_period[0], yearfirst=False)
    end = start + pd.Timedelta(simulation_period[1], unit="D")
    time_slice = timeseries[start:end]

    esys = create_solph_model(
        techparam=tech_param,
        timeindex=time_slice.index,
        timeseries=time_slice,
    )
    esys = solve_model(esys)

    # print and plot some results
    results = esys.results["main"]

    heat_gen = solph.views.node(results, "heat_generation")
    heat_store = solph.views.node(results, "thermal_storage")
    elec = solph.views.node(results, "electricity")

    print(heat_gen["sequences"].sum())
    print(heat_store["sequences"].sum())
    print(elec["sequences"].sum())

    fig1, ax = plt.subplots(figsize=(10, 5))
    heat_gen["sequences"].plot(ax=ax)
    plt.show()

    fig2, ax = plt.subplots(figsize=(10, 5))
    heat_store["sequences"].plot(ax=ax)
    plt.show()

    fig3, ax = plt.subplots(figsize=(10, 5))
    elec["sequences"].plot(ax=ax)
    plt.show()

    # ################################

    # what is the minimum possible emission value?
    esys_emission = create_solph_model(
        techparam=tech_param,
        timeindex=time_slice.index,
        timeseries=time_slice,
        weight_cost_emission=1,
    )
    esys_emission = solve_model(esys_emission)
    emission_min = esys_emission.results["meta"]["objective"]  # emission in [kg]

    # what is the emission value in the cost optimal case?
    esys_cost = create_solph_model(
        techparam=tech_param,
        timeindex=time_slice.index,
        timeseries=time_slice,
        weight_cost_emission=0,
    )

    esys_max = solve_model(esys_cost)
    emission_max = esys_max.results["meta"]["emission_value"]
    cost_max = esys_max.results["meta"]["objective"]

    esys_min = solve_model(esys_cost, emission_limit=emission_min + 0.1)
    costs_min = esys_min.results["meta"]["objective"]

    factor_emission_reduction = 0.5
    emission_limit_mid = \
        factor_emission_reduction * (emission_max - emission_min) + emission_min
    esys_mid = solve_model(esys_cost, emission_limit=emission_limit_mid)
    emission_mid = esys_mid.results["meta"]["emission_value"]
    costs_mid = esys_mid.results["meta"]["objective"]

    # plots
    fig, ax = plt.subplots()
    ax.scatter(emission_max, cost_max, color='r')
    ax.scatter(emission_min, costs_min, color='b')
    ax.scatter(emission_mid, costs_mid, color='tab:orange')
    ax.set_xlabel('Emission [kg]')
    ax.set_ylabel('Costs [€]')
    ax.grid()
    # ax.set_title('scatter plot')
    plt.show()

    for es in [esys_min, esys_max, esys_mid]:
        plot_results(es)
        results = es.results["main"]
        b_elec = solph.views.node(results, "electricity")[
            "sequences"].sum()
        print(" ")
        print(b_elec)


    # #####

    heat = []
    elec = []
    gas = []
    h2 = []
    for es in [esys_min, esys_mid, esys_max]:
        results = es.results["main"]

        heat_gen = solph.views.node(results, "heat_generation")[
            "sequences"].sum()
        heat.append(heat_gen)

        elec_sum = solph.views.node(results, "electricity")[
            "sequences"].sum()
        elec.append(elec_sum)

        h2_sum = solph.views.node(results, "h2")[
            "sequences"].sum()
        h2.append(h2_sum)

        gas_sum = solph.views.node(results, "gas")[
            "sequences"].sum()
        gas.append(gas_sum)

    df_heat_all = pd.concat(heat, axis=1)
    df_heat_all.columns = ["CO2_min", "CO2_mid", "CO2_max"]
    # df_heat_all = df_heat_all.T

    df_elec = pd.concat(elec, axis=1)
    df_elec.columns = ["CO2_min", "CO2_mid", "CO2_max"]
    # df_elec = df_elec.T

    df_gas = pd.concat(gas, axis=1)
    df_gas.columns = ["CO2_min", "CO2_mid", "CO2_max"]
    # df_gas = df_gas.T

    df_h2 = pd.concat(h2, axis=1)
    df_h2.columns = ["CO2_min", "CO2_mid", "CO2_max"]
    # df_h2 = df_h2.T

    fig, ax = plt.subplots()
    df_heat_all.plot.bar(stacked=True)
    plt.show()

    logging.info("Done!")
