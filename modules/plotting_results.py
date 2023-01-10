import matplotlib.pyplot as plt
import pandas as pd
import parse
import plotly.graph_objects as go
import modules.res_tools_flexible as res
import numpy as np
import os

filenames = ["results/data/results_Scenario A_ErrorProfiles_input.csv",
             "results/data/results_Scenario A_ErrorProfiles_input_Boiler_14_10_18.csv",
             "results/data/results_Scenario A_ErrorProfiles_input_CHP_13_1_18.csv",
             "results/data/results_Scenario B_ErrorProfiles_input.csv",
             "results/data/results_Scenario B_ErrorProfiles_input_Boiler_14_10_18.csv",
             "results/data/results_Scenario B_ErrorProfiles_input_CHP_13_1_18.csv",
             "results/data/results_Scenario C_ErrorProfiles_input.csv",
             "results/data/results_Scenario C_ErrorProfiles_input_Boiler_14_10_18.csv",
             "results/data/results_Scenario C_ErrorProfiles_input_CHP_13_1_18.csv"]

vars = ['controller.calc_Qdot_production.u_Qdot_Boiler',
        'controller.calc_Qdot_production.u_Qdot_CHP',
        'controller.calc_Qdot_production.u_Qdot_Electrolyzer',
        'controller.calc_Qdot_production.u_Qdot_Heatpump1',
        'controller.calc_Qdot_production.u_Qdot_Heatpump2',
        'controller.calc_Qdot_production.y_Qdot',
        'dynamic_Heatload_Scale.Qdot_heatload_scaled',
        'dynamic_Heatload_Scale.Qdot_heatload',
        'fMU_PhyModel.temperature_HeatGrid_FF.T',
        "controller.u_T_HeatGrid_FF_set"
        ]

def temperature_control(scenarios=["A", "B", "C"], errors=["Boiler_14_10_18", "CHP_13_1_18"],
                            temp_var="fMU_PhyModel.temperature_HeatGrid_FF.T",
                            temp_set="controller.u_T_HeatGrid_FF_set"):
    """Plots a figure with a subplot for each scenario, where the temperatures for each error file are compared with
    each other and the set temperature.The last subplot shows all temperature variables for all scenarios together.

    Arguments:
        scenarios: list of strings with the names or distinction between scenarios. Default: ["A", "B", "C"].
        errors: list of strings with the names or distinction between errors. Default: ["Boiler_14_10_18",
        "CHP_13_1_18"].
        temp_var: string. Name of the temperature variable in the model. Default:
        "fMU_PhyModel.temperature_HeatGrid_FF.T".
        temp_set: string. Name of the set temperature variable in the model. Default: "controller.u_T_HeatGrid_FF_set".
        """

    #if not os.path.isdir("data"):
    #    os.mkdir("data")

    os.chdir("results/data")
    files_list = os.listdir()
    csv_list = []
    for file in files_list:
        if ".CSV" in file:
            csv_list.append(file)

    scenarios_dict = {}
    for scenario in scenarios:
        temp_dict = {}
        for element in csv_list:
            if scenario in element:
                df = pd.read_csv(element)
                df[temp_var] = df[temp_var] - 273.15
                info = scenario
                temp_dict[info] = df[temp_var]
                temp_dict[scenario + "_set"] = df[temp_set]
                for error in errors:
                    if error in element:
                        info = info + "_" + error

                temp_dict[info] = df[temp_var]

        scenarios_dict[scenario] = temp_dict

    fig, axs = plt.subplots(4, 1, figsize=(15, 8))

    colors = {}
    colors[0] = ["b-", "b--", "b:", "r-"]
    colors[1] = ["g-", "g--", "g:", "r-"]
    colors[2] = ["c-", "c--", "c:", "r-"]

    for count, scenario in enumerate(scenarios):
        for key_count, key in enumerate(scenarios_dict[scenario].keys()):
            axs[count].plot(df["time"], scenarios_dict[scenario][key], colors[count][key_count])
            axs[count].set_title("Scenario " + scenario)
            axs[count].set_ylabel("Temperature (C)")

    for count, scenario in enumerate(scenarios):
        for key_count, key in enumerate(scenarios_dict[scenario].keys()):
            split_key = key.split("_")
            label = "scenario " + split_key[0]
            if error[0] in key:
                label = "T with boiler error, " + label
            elif error[1] in key:
                label = "T with chp error, " + label
            elif "set" in key:
                label = "T set, " + label
            else:
                label = "T " + label
            axs[3].plot(df["time"], scenarios_dict[scenario][key], colors[count][key_count], label=label)
    axs[3].set_title("All scenarios")
    axs[3].set_ylabel("Temperature (C)")
    axs[3].set_xlabel("Time (min)")

    fig.legend()

    fig.tight_layout()
    plt.savefig("results/data/temperature_control.png")

def resilience_box_plot(data_file="results/data/resilience.csv", scenarios=["A", "B", "C"],
                        errors=["Boiler_14_10_18", "CHP_13_1_18"]):
    """Saves a box plot of the resilience indices for the different scenarios, considering several possible errors, and
    the corresponding values are saved as a csv file as well.

    Arguments:
        data_file: Name of the csv file with the resilience information (string). Default: "data/resilience.csv"
        scenarios: List of the possible scenarios (list of strings). Default: ["A", "B", "C"].
        errors: List of the names of the considered errors (list of strings).
        Default: ["Boiler_14_10_18", "CHP_13_1_18"]
        """
    data = pd.read_csv(data_file)
    RI_values = {}
    for s in scenarios:
        error_dict = {}
        for column_name in data.columns:
            if s in column_name:
                key = ""
                for error in errors:
                    if error in column_name:
                        key = error
                if not key:
                    key = "No-error"
                error_dict[key] = data.at[3, column_name]

        RI_values[s] = error_dict
    RI_df = pd.DataFrame(data=RI_values)

    scenarios_data = []
    xticks_labels = []
    xticks = []

    for ii in range(len(scenarios)):
        scenarios_data.append(RI_df[scenarios[ii]])
        xticks_labels.append("Scenario " + scenarios[ii])
        xticks.append(ii+1)

    fig, ax = plt.subplots()
    ax.boxplot(scenarios_data)
    plt.xticks(xticks, xticks_labels, rotation=10)
    plt.ylabel("Resilience Index")
    plt.title("Resilience Index")
    plt.tight_layout()
    plt.savefig("resilience_boxplot.png")

    # And we also save the new data frame with averages as well
    #RI_df.loc[len(RI_df.index)] = [RI_df["A"].mean(), RI_df["B"].mean(), RI_df["C"].mean()]
    #RI_df.index = ["No-error", "Boiler_14_10_18", "CHP_13_1_18", "Average"]
    #RI_df.to_csv("Scenarios_resilience_w_average.csv")


def separate_plots(filename="results/data/results_Scenario A_ErrorProfiles_input.csv", vars=vars,
                   scenarios=["A", "B", "C"]):
    """Saves separate plots for the heatload (with and without scaling), the temperature (real and set temperatures) and
     the production of the different generators.

     Arguments:
        filename: Name of the csv file with the results (string)
        vars: List of the output variable names (list of strings)
        scenarios: List of the possible scenarios (list of strings). Default: ["A", "B", "C"].
         """
    title = ""
    for scenario in scenarios:
        if scenario in filename:
            title = "Scenario_" + scenario

    error = parse.search('ErrorProfiles_input{}.csv', filename)
    if error:
        title = title + " with error" + error.fixed[0].replace("_", " ", 2).replace("_", "-")

    data = pd.read_csv(filename)
    data["fMU_PhyModel.temperature_HeatGrid_FF.T"] = data["fMU_PhyModel.temperature_HeatGrid_FF.T"] - 273.15

    # HEATLOAD PLOT
    plt.clf()
    plt.plot(data["time"], data[vars[6]], label="Heatload")
    plt.plot(data["time"], data[vars[7]], linestyle="-", label="Scaled Heatload")
    plt.legend()
    plt.title("Heatload", fontsize=14)
    plt.xlabel("Time (min)")
    plt.ylabel("Power (kW)")
    plt.tight_layout()
    plt.savefig("heatload_plot_" + title.replace(" ", "_") + ".png")

    # TEMPERATURE PLOT
    plt.clf()
    plt.plot(data["time"], data[vars[8]], label="T")
    plt.plot(data["time"], data[vars[9]], label="set T")
    plt.legend()
    plt.xlabel("Time (min)")
    plt.ylabel("Temperature (C)")
    plt.title("Temperature", fontsize=14)
    plt.tight_layout()
    plt.savefig("temperature_plot_" + title.replace(" ", "_") + ".png")

    # PRODUCTION PLOT
    plt.clf()
    plt.plot(data["time"], data[vars[0]], label="Boiler")

    plt.plot(data["time"], data[vars[1]], label="CHP")

    plt.plot(data["time"], data[vars[2]], label="Electrolyzer")

    plt.plot(data["time"], data[vars[3]], label="Heat Pump 1")
    plt.plot(data["time"], data[vars[4]], label="Heat Pump 2")

    plt.plot(data["time"], data[vars[5]], label="Qdot")
    plt.xlabel("Time (min)")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.title("Production", fontsize=14)

    plt.tight_layout()
    plt.savefig("production_plot" + title.replace(" ", "_") + ".png")
    plt.clf()

def plot(data_file="results/data/results_Scenario A_ErrorProfiles_input.csv",
         vars=vars, scenarios=["A", "B", "C"]):
    """
    Function that plots every output variable in a separate subplot for a specific scenario and
     saves the plot as a png file with a name indicating the parameter scenario and error file
     that were used to generate the data file.

    Arguments:
        data_file: Name of the csv file with the results (string)
        vars: List of the output variable names (list of strings)
        scenarios: List of the possible dimension_scenarios (list of strings). Default: ["A", "B", "C"].
    """

    data = pd.read_csv(data_file)
    data["fMU_PhyModel.temperature_HeatGrid_FF.T"] = data["fMU_PhyModel.temperature_HeatGrid_FF.T"] - 273.15
    fig, axs = plt.subplots(3, figsize=(12, 8))

    # filtering the scenario the data belongs to and setting it as pic title
    title = data_file.split("/")[1]
    # for s in dimension_scenarios:
    #     if s in data_file:
    #         title = "Scenario " + s

    # looking for the corresponding error file used and adding it to the pic title
    error = parse.search('ErrorProfiles_input{}.csv', data_file)
    if error:
        title = title + " with error" + error.fixed[0].replace("_", " ", 2).replace("_", "-")

    fig.suptitle(title, fontsize=18)

    axs[0].plot(data["time"], data[vars[0]], label="Boiler")
    axs[0].plot(data["time"], data[vars[1]], label="CHP")
    axs[0].plot(data["time"], data[vars[2]], label="Electrolyzer")
    axs[0].plot(data["time"], data[vars[3]], label="Heat Pump 1")
    axs[0].plot(data["time"], data[vars[4]], label="Heat Pump 2")

    axs[0].set_title("Production", fontsize=14)
    axs[0].legend()

    axs[1].plot(data["time"], data[vars[5]], label="Qdot")
    axs[1].plot(data["time"], data[vars[7]], label="Heatload")
    axs[1].plot(data["time"], data[vars[6]], linestyle="-", label="Scaled Heatload")
    axs[1].set_title("Heatload", fontsize=14)
    axs[1].legend()

    axs[2].plot(data["time"], data[vars[8]], label="T")
    axs[2].plot(data["time"], data[vars[9]], label="set T")
    axs[2].legend()
    axs[2].set_title("Temperature", fontsize=14)

    fig.tight_layout()

    plt.savefig("results/data/" + title.replace(" ", "_") + ".png")


def getting_data_radar_chart():
    load = 290  # (aprox.) from excel file "Quarree100_load_15_Modelica"

    redundancy_list = []
    stirling_list = []
    shannon_list = []

    for sheet in ["Anlagen_basic",
                  "Anlagen_examples_for_p_inst",
                  "Anlagen_more_indices",
                  "Anlagen_more_indices_2"]:
        l_o_s = res.summon_systems(path_to_excel_file="examples/excel_data/res_tools_example_data.xlsx",
                                   sheet_name=sheet)
        redundancy_list.append(res.redundancy(load, l_o_s, "p_inst_out_th"))
        shannon_list.append(res.shannon_index(l_o_s, "p_inst_out_th"))
        stirling_list.append(res.stirling_index(l_o_s, "p_inst_out_th"))

    redundancy = np.array(redundancy_list)
    shannon = np.array(shannon_list)
    stirling = np.array(stirling_list)

    redundancy_transformed = np.zeros(4)
    max_val = redundancy.max() + redundancy.mean()/5
    min_val = redundancy.min() - redundancy.mean()/5
    val_range = max_val - min_val
    for count, element in enumerate(redundancy):
        redundancy_transformed[count] = (element - min_val) * 100 / val_range

    stirling_transformed = np.zeros(4)
    max_val = stirling.max() + stirling.mean()/5
    min_val = stirling.min() - stirling.mean()/5
    val_range = max_val - min_val
    for count, element in enumerate(stirling):
        stirling_transformed[count] = (element - min_val) * 100 / val_range

    shannon_transformed = np.zeros(4)
    max_val = shannon.max() + shannon.mean()/5
    min_val = shannon.min() - shannon.mean()/5
    val_range = max_val - min_val
    for count, element in enumerate(shannon):
        shannon_transformed[count] = (element - min_val) * 100 / val_range

    return shannon_transformed, stirling_transformed, redundancy_transformed

def radar_chart(scenarios = ["Scenario A", "Scenario B", "Scenario C", "Scenario D"]):

    categories = ["Shannon Index", "Stirling Index", "Redundancy"]

    shannon, stirling, redundancy = getting_data_radar_chart()

    fig = go.Figure()

    for ii in range(len(scenarios)):
        fig.add_trace(go.Scatterpolar(
            r=[shannon[ii], stirling[ii], redundancy[ii]],
            theta=categories,
            fill='toself',
            name=scenarios[ii]
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True
    )

    #fig.show()

    fig.write_image("radar_chart_exp.png")  # specifically kaleido v0.1.0.post1 was required for this line

