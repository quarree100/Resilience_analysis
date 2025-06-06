import matplotlib.pyplot as plt
import pandas as pd
import parse
import plotly.graph_objects as go
#import modules.res_tools_flexible as res
#import numpy as np
from kaleido.scopes.plotly import PlotlyScope
from matplotlib.backends.backend_pdf import PdfPages
import os


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
                            temp_set="controller.u_T_HeatGrid_FF_set", store_results=None):

    """Plots a figure with a subplot for each scenario, where the temperatures for each error file are compared with
    each other and the set temperature.The last subplot shows all temperature variables for all scenarios together.

    Arguments:
        scenarios: list of strings with the names or distinction between scenarios. Default: ["A", "B", "C"].
        errors: list of strings with the names or distinction between errors. Default: ["Boiler_14_10_18",
        "CHP_13_1_18"].
        temp_var: string. Name of the temperature variable in the model. Default:
        "fMU_PhyModel.temperature_HeatGrid_FF.T".
        temp_set: string. Name of the set temperature variable in the model. Default: "controller.u_T_HeatGrid_FF_set".
        store_results: String of the path where the data is to be found and the plots to be saved.
         Default: None.
        """

    files_list = os.listdir(os.path.join(store_results, "data"))
    csv_list = []
    for file in files_list:
        if "results" in file:
            csv_list.append(file)

    scenarios_dict = {}
    for scenario in scenarios:
        temp_dict = {}
        for element in csv_list:
            if scenario in element:
                df = pd.read_csv(os.path.join(store_results, "data", element))
                df[temp_var] = df[temp_var] - 273.15
                #info = scenario
                for error in errors:
                    if error in element:
                        info = error

                temp_dict[info] = df[temp_var]
                temp_dict[info + "_set"] = df[temp_set]

        scenarios_dict[scenario] = temp_dict

    fig, axs = plt.subplots(len(scenarios)+1, 1, figsize=(15, 8))

    #colors = {}
    #colors[0] = ["b-", "b--", "b:", "r-"]
    #colors[1] = ["g-", "g--", "g:", "r-"]
    #colors[2] = ["c-", "c--", "c:", "r-"]

    cases = scenarios_dict[scenario].keys()  # a given scenario and a given error = case

    for count, scenario in enumerate(scenarios):
        axs[count].set_title(scenario)
        axs[count].set_ylabel("Temperature (C)")
        for key_count, key in enumerate(cases):
            label = scenario
            for error in errors:
                if error in key:
                    if "reference" in error:
                        label = label + " reference"
                        color = "g"
                        label_scenario = "reference"
                    else:
                        label = label + " with error " + error
                    if "short" in label:
                        color = "r"
                        label_scenario = "short duration error"
                    elif "medium" in label:
                        color = "b"
                        label_scenario = "medium duration error"
                    elif "long" in label:
                        color = "y"
                        label_scenario = "long duration error"
            if "set" in key:
                label = "T set " + label
                label_scenario = "T set " + label_scenario
            else:
                label = "T " + label
            axs[count].plot(df["time"], scenarios_dict[scenario][key], color, label=label_scenario)  # , colors[count][key_count])
            axs[count].legend()
            axs[len(scenarios)].plot(df["time"], scenarios_dict[scenario][key], label=scenario)

    if False:
        for count, scenario in enumerate(scenarios):
            for key_count, key in enumerate(cases):
                label = scenario
                for error in errors:
                    if error in key:
                        if "reference" in error:
                            label = label + " reference"
                        else:
                            label = label + " with error " + error
                if "set" in key:
                    label = "T set " + label
                else:
                    label = "T " + label
                axs[len(scenarios)].plot(df["time"], scenarios_dict[scenario][key], label=label)
    axs[len(scenarios)].set_title("All scenarios")
    axs[len(scenarios)].set_ylabel("Temperature (C)")
    axs[len(scenarios)].set_xlabel("Time (min)")
    axs[len(scenarios)].legend()

    #fig.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(store_results, "plots", "temperature_control.png"))

def resilience_box_plot(data_file="results/data/resilience.csv", scenarios=["A", "B", "C"],
                        errors=["Boiler_14_10_18", "CHP_13_1_18"], store_results=None):
    """Saves a box plot of the resilience indices for the different scenarios, considering several possible errors, and
    the corresponding values are saved as a csv file as well.

    Arguments:
        data_file: Name of the csv file with the resilience information (string). Default: "data/resilience.csv"
        scenarios: List of the possible scenarios (list of strings). Default: ["A", "B", "C"].
        errors: List of the names of the considered errors (list of strings).Default: ["Boiler_14_10_18",
         "CHP_13_1_18"]
        store_results: String of the path where the data is to be found and the plots to be saved.
         Default: None.
        """
    data = pd.read_csv(os.path.join(store_results, "data", data_file))
    RI_values = {}
    for s in scenarios:
        error_dict = {}
        for column_name in data.columns:
            if s in column_name:
                key = ""
                for error in errors:
                    if error in column_name:
                        key = error
                #if not key:
                #    key = "No-error"
                error_dict[key] = data.at[3, column_name]

        RI_values[s] = error_dict
    RI_df = pd.DataFrame(data=RI_values)

    scenarios_data = []
    xticks_labels = []
    xticks = []

    for ii in range(len(scenarios)):
        scenarios_data.append(RI_df[scenarios[ii]])
        xticks_labels.append(scenarios[ii])
        xticks.append(ii+1)

    fig, ax = plt.subplots()
    ax.boxplot(scenarios_data)
    plt.xticks(xticks, xticks_labels, rotation=10)
    plt.ylabel("Resilience Index")
    plt.title("Resilience Index")
    plt.tight_layout()
    plt.savefig(os.path.join(store_results, "plots", "resilience_boxplot.png"))

    # And we also save the new data frame with averages as well
    #RI_df.loc[len(RI_df.index)] = [RI_df["A"].mean(), RI_df["B"].mean(), RI_df["C"].mean()]
    #RI_df.index = ["No-error", "Boiler_14_10_18", "CHP_13_1_18", "Average"]
    #RI_df.to_csv("Scenarios_resilience_w_average.csv")


def separate_plots(filename="results_Scenario A_ErrorProfiles_input.csv", vars=vars,
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
    plt.plot(data["time"], data[vars[6]], label="Scaled Heatload")
    plt.plot(data["time"], data[vars[7]], linestyle="-", label="Heatload")
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


def plot(data_file="results_Scenario A_ErrorProfiles_input.csv", store_results=None,
         vars=vars, scenarios=["A", "B", "C"]):
    """
    Function that plots every output variable in a separate subplot for a specific scenario and
     saves the plot as a png file with a name indicating the parameter scenario and error file
     that were used to generate the data file.

    Arguments:
        data_file: Name of the csv file with the results (string)
        store_results: String of the path where the data is to be found and the plots to be saved.
         Default: None.
        vars: List of the output variable names (list of strings)
        scenarios: List of the possible dimension_scenarios (list of strings). Default: ["A", "B", "C"].
    """

    data = pd.read_csv(os.path.join(store_results, "data", data_file))
    data["fMU_PhyModel.temperature_HeatGrid_FF.T"] = data["fMU_PhyModel.temperature_HeatGrid_FF.T"] - 273.15
    fig, axs = plt.subplots(3, figsize=(12, 8))

    # filtering the scenario the data belongs to and setting it as pic title
    title = ""
    for s in scenarios:
        if s in data_file:
            title = s
            break

    # looking for the corresponding error file used and adding it to the pic title
    data_file.strip(".csv")
    error = data_file.strip(f"results_{title}_")

    subtitle = " with error " + error.replace("_", " ")
    if ".csv" in subtitle:
        subtitle = subtitle.strip(".csv")

    if "reference" in error:
        subtitle = " reference"

    #if "reference" not in error:
        #title = title + " with error" + error.fixed[0].replace("_", " ", 2).replace("_", "-")

    fig.suptitle(title + subtitle, fontsize=18)

    axs[0].plot(data["time"], data[vars[0]], label="Boiler")
    axs[0].plot(data["time"], data[vars[1]], label="CHP")
    axs[0].plot(data["time"], data[vars[2]], label="Electrolyzer")
    axs[0].plot(data["time"], data[vars[3]], label="Heat Pump 1")
    axs[0].plot(data["time"], data[vars[4]], label="Heat Pump 2")

    axs[0].set_title("Production", fontsize=14)
    axs[0].legend()

    axs[1].plot(data["time"], data[vars[5]], label="Qdot")
    #axs[1].plot(data["time"], data[vars[6]], linestyle="-", label="Scaled Heatload")
    axs[1].plot(data["time"], data[vars[7]], label="Heatload")
    axs[1].set_title("Heatload", fontsize=14)
    axs[1].legend()

    axs[2].plot(data["time"], data[vars[8]], label="T")
    axs[2].plot(data["time"], data[vars[9]], label="set T")
    axs[2].legend()
    axs[2].set_title("Temperature", fontsize=14)

    fig.tight_layout()

    plot_name = title.replace(" ", "_") + subtitle.replace(" ", "_") + ".png"
    plt.savefig(os.path.join(store_results, "plots", plot_name))

def radar_chart(store_results, attributes, scenarios, categories):
    """
    Makes the radar chart plot of the resilience attributes: Shannon index, Stirling index and redundancy, for each
    scenario and saves it in the "store_results" path.

    Args:
        store_results: string. Path where the output will be saved.
        attributes: list of arrays where each array is one of the attributes and each element corresponds to a
        scenario.
        scenarios: list of strings. Names of the scenarios considered.
        categories: list of strings. Names of the attributes.

    """
    # separation of the attributes for convenience
    shannon = attributes[0]
    stirling = attributes[1]
    redundancy = attributes[2]
    fig = go.Figure()

    for count, scenario in enumerate(scenarios):
        fig.add_trace(go.Scatterpolar(
            r=[shannon[count], stirling[count], redundancy[count]],
            theta=categories,
            fill='toself',
            opacity=0.5,
            name=scenario
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.2]
            )),
        showlegend=True
    )
    #fig.show()
    plot_path = os.path.join(store_results, "plots", "radar_chart.png")
    fig.write_image(plot_path)  # specifically kaleido v0.1.0.post1 was required for this line


def plot_combined_wo_seasons(scenario_list=["2020_emission-low", "2020_emission-high", "2030_syn-gas-high_emission-low",
                                            "2030_syn-gas-high_emission-high", "2030_syn-gas-low_emission-low",
                                            "2030_syn-gas-low_emission-high"],
                             duration=["short duration (3h)", "medium duration (1d)", "long duration (7d)", "combined"],
                             season=["transitional", "summer", "winter"],
                             folder_path=r'C:\Users\pooji\forunilaptop\artec\Untitled Folder',
                             save_as_pdf=False):  # Add a save_as_pdf argument

    """

This code works WITHOUT season functionality.
It generates combined plots for each scenario irrespective of the seasons. (To get this, DO NOT enter duration argument while calling the function)
It also generates individual plots when duration argument is entered.
If the folder has all files of the same season, then it gives correct combined plots
Beware! it also works for files that are of different seasons and considers them for plotting

Folder path will be passed as the current results folder when the function is called from calc_scenarios.

    """

    printed_count = 0
    scenario_dfs = {}  # Dictionary to store data frames for each scenario

    # Initialize PdfPages object with the full path to the PDF file
    pdf_path = os.path.join(folder_path, 'plots.pdf')
    pdf_pages = PdfPages(pdf_path)  # Create a PDF file for saving plots

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if not file_name.endswith("reference.csv"):
                for scenario in scenario_list:
                    for dur in duration:  # Iterate through specified durations
                        if scenario in file_name and dur in file_name:  # Check both scenario and duration
                            for s in season:
                                if s in file_name:
                                    file_path = os.path.join(root, file_name)
                                    df = pd.read_csv(file_path,
                                                     usecols=["time", "fMU_PhyModel.temperature_HeatGrid_FF.T",
                                                              "controller.u_T_HeatGrid_FF_set"])

                                    # Convert Kelvin to Celsius for "fMU_PhyModel.temperature_HeatGrid_FF.T"
                                    df["fMU_PhyModel.temperature_HeatGrid_FF.T"] -= 273.15

                                    # Create a unique column name based on the file name
                                    col_name = os.path.splitext(file_name)[0]  # Remove the file extension

                                    # Rename the temperature column
                                    df.rename(columns={"fMU_PhyModel.temperature_HeatGrid_FF.T": col_name},
                                              inplace=True)

                                    # Merge data frames
                                    if scenario not in scenario_dfs:
                                        scenario_dfs[scenario] = df[
                                            ["time", "controller.u_T_HeatGrid_FF_set", col_name]]
                                    else:
                                        scenario_dfs[scenario] = scenario_dfs[scenario].merge(df[["time", col_name]],
                                                                                              on="time", how="outer")

                                    printed_count += 1

    print(f"Total data frames printed: {printed_count}")

    # Calculate minimum, maximum, and average for each type of "duration" for each scenario
    for scenario in scenario_list:
        for dur in duration:
            if scenario in scenario_dfs:
                scenario_dur_columns = [col for col in scenario_dfs[scenario].columns if dur in col]
                if scenario_dur_columns:
                    scenario_dfs[scenario][f"{dur}_Minimum"] = scenario_dfs[scenario][scenario_dur_columns].min(axis=1)
                    scenario_dfs[scenario][f"{dur}_Maximum"] = scenario_dfs[scenario][scenario_dur_columns].max(axis=1)
                    scenario_dfs[scenario][f"{dur}_Average"] = scenario_dfs[scenario][scenario_dur_columns].mean(axis=1)

    # Create plots for each scenario
    if not duration or "combined" in duration:
        for idx, (scenario, df) in enumerate(scenario_dfs.items()):
            plt.figure(figsize=(12, 6))
            plt.suptitle(f"Scenario: {scenario}", fontsize=16)  # Add main title for each scenario

            # Plot 1: Short Duration
            if not duration or "short duration (3h)" in duration:
                plt.subplot(2, 2, 1)
                plt.plot(df["time"], df["controller.u_T_HeatGrid_FF_set"], label="Controller Temperature",
                         color="black")
                plt.plot(df["time"], df[f"short duration (3h)_Minimum"], label="Short Duration Minimum", color="grey")
                plt.plot(df["time"], df[f"short duration (3h)_Maximum"], label="Short Duration Maximum", color="grey")
                plt.plot(df["time"], df[f"short duration (3h)_Average"], label="Short Duration Average", color="grey",
                         linestyle="--")
                plt.fill_between(df["time"], df[f"short duration (3h)_Minimum"], df[f"short duration (3h)_Maximum"],
                                 color="lightgrey")
                plt.title("Short Duration")
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.xlabel("Time")
                plt.ylabel("Temperature (Celsius)")

            # Plot 2: Medium Duration
            if not duration or "medium duration (1d)" in duration:
                plt.subplot(2, 2, 2)
                plt.plot(df["time"], df["controller.u_T_HeatGrid_FF_set"], label="Controller Temperature",
                         color="black")
                plt.plot(df["time"], df[f"medium duration (1d)_Minimum"], label="Medium Duration Minimum", color="grey")
                plt.plot(df["time"], df[f"medium duration (1d)_Maximum"], label="Medium Duration Maximum", color="grey")
                plt.plot(df["time"], df[f"medium duration (1d)_Average"], label="Medium Duration Average", color="grey",
                         linestyle="--")
                plt.fill_between(df["time"], df[f"medium duration (1d)_Minimum"], df[f"medium duration (1d)_Maximum"],
                                 color="lightgrey")
                plt.title("Medium Duration")
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.xlabel("Time")
                plt.ylabel("Temperature (Celsius)")

            # Plot 3: Long Duration
            if not duration or "long duration (7d)" in duration:
                plt.subplot(2, 2, 3)
                plt.plot(df["time"], df["controller.u_T_HeatGrid_FF_set"], label="Controller Temperature",
                         color="black")
                plt.plot(df["time"], df[f"long duration (7d)_Minimum"], label="Long Duration Minimum", color="grey")
                plt.plot(df["time"], df[f"long duration (7d)_Maximum"], label="Long Duration Maximum", color="grey")
                plt.plot(df["time"], df[f"long duration (7d)_Average"], label="Long Duration Average", color="grey",
                         linestyle="--")
                plt.fill_between(df["time"], df[f"long duration (7d)_Minimum"], df[f"long duration (7d)_Maximum"],
                                 color="lightgrey")
                plt.title("Long Duration")
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.xlabel("Time")
                plt.ylabel("Temperature (Celsius)")

            # Plot 4: Combined Durations (Only generated if "combined" is in duration)
            if not duration or "combined" in duration:
                plt.subplot(2, 2, 4)
                plt.plot(df["time"], df["controller.u_T_HeatGrid_FF_set"], label="Controller Temperature",
                         color="black")
                plt.plot(df["time"], df[f"short duration (3h)_Minimum"], label="Short Duration Minimum", color="grey")
                plt.plot(df["time"], df[f"short duration (3h)_Maximum"], label="Short Duration Maximum", color="grey")
                plt.plot(df["time"], df[f"short duration (3h)_Average"], label="Short Duration Average", color="grey",
                         linestyle="--")
                plt.plot(df["time"], df[f"medium duration (1d)_Minimum"], label="Medium Duration Minimum", color="grey")
                plt.plot(df["time"], df[f"medium duration (1d)_Maximum"], label="Medium Duration Maximum", color="grey")
                plt.plot(df["time"], df[f"medium duration (1d)_Average"], label="Medium Duration Average", color="grey",
                         linestyle="--")
                plt.plot(df["time"], df[f"long duration (7d)_Minimum"], label="Long Duration Minimum", color="grey")
                plt.plot(df["time"], df[f"long duration (7d)_Maximum"], label="Long Duration Maximum", color="grey")
                plt.plot(df["time"], df[f"long duration (7d)_Average"], label="Long Duration Average", color="grey",
                         linestyle="--")
                plt.fill_between(df["time"], df[f"short duration (3h)_Minimum"], df[f"long duration (7d)_Maximum"],
                                 color="lightgrey")
                plt.title(f"{scenario} - Combined Durations")
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.xlabel("Time")
                plt.ylabel("Temperature (Celsius)")

            plt.tight_layout()

            if save_as_pdf:
                pdf_pages.savefig()  # Save the current figure as a page in the PDF

            plt.show()

    elif duration:
        for scenario, df in scenario_dfs.items():
            for dur in duration:
                plt.figure(figsize=(12, 6))
                plt.title(f"Scenario: {scenario} - Duration: {dur}",
                          fontsize=16)  # Add title for each scenario and duration

                # Plot Controller Temperature in black
                plt.plot(df["time"], df["controller.u_T_HeatGrid_FF_set"], label="Controller Temperature",
                         color="black")

                # Plot Duration with min, max, and average
                plt.plot(df["time"], df[f"{dur}_Minimum"], label=f"{dur} Minimum", color="grey")
                plt.plot(df["time"], df[f"{dur}_Maximum"], label=f"{dur} Maximum", color="grey")
                plt.plot(df["time"], df[f"{dur}_Average"], label=f"{dur} Average", color="grey", linestyle="--")

                # Fill region between min and max with light grey
                plt.fill_between(df["time"], df[f"{dur}_Minimum"], df[f"{dur}_Maximum"], color="lightgrey")

                plt.legend()
                plt.xlabel("Time")
                plt.ylabel("Temperature (Celsius)")
                plt.tight_layout()

                if save_as_pdf:
                    pdf_pages.savefig()  # Save the current figure as a page in the PDF

                plt.show()

    if save_as_pdf:
        pdf_pages.close()  # Close the PDF after saving all figures

    return scenario_dfs


def plot_seasons_included(scenario_list=["2020_emission-low", "2020_emission-high"],
         duration=["short duration (3h)", "medium duration (1d)", "long duration (7d)", "combined"],
         season=["transitional", "summer", "winter", "all"],
         folder_path=r'C:\Users\pooji\forunilaptop\artec\to test seasons',
         save_as_pdf=False):

    """

This code works WITH season functionality.
It DOESN'T generate combined plots for each scenario irrespective of the seasons.
It olny generates the individual plots

Folder path will be passed as the current results folder when the function is called from calc_scenarios.


    """

    printed_count = 0
    scenario_dfs = {}

    pdf_pages = None

    if save_as_pdf:
        pdf_pages = PdfPages('Plots.pdf')

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if not file_name.endswith("reference.csv"):
                for scenario in scenario_list:
                    for dur in duration:
                        for s in season:
                            if scenario in file_name and dur in file_name and s in file_name:
                                file_path = os.path.join(root, file_name)
                                df = pd.read_csv(file_path, usecols=["time", "fMU_PhyModel.temperature_HeatGrid_FF.T",
                                                                     "controller.u_T_HeatGrid_FF_set"])

                                df["fMU_PhyModel.temperature_HeatGrid_FF.T"] -= 273.15

                                col_name = os.path.splitext(file_name)[0]

                                df.rename(columns={"fMU_PhyModel.temperature_HeatGrid_FF.T": col_name}, inplace=True)

                                if scenario not in scenario_dfs:
                                    scenario_dfs[scenario] = {}
                                if s not in scenario_dfs[scenario]:
                                    scenario_dfs[scenario][s] = {}
                                if dur not in scenario_dfs[scenario][s]:
                                    scenario_dfs[scenario][s][dur] = df[
                                        ["time", "controller.u_T_HeatGrid_FF_set", col_name]]
                                else:
                                    scenario_dfs[scenario][s][dur] = scenario_dfs[scenario][s][dur].merge(
                                        df[["time", col_name]], on="time", how="outer")

                                printed_count += 1

    # print(f"Total data frames printed: {printed_count}")

    # Calculate minimum, maximum, and average values
    for scenario in scenario_list:
        for s in season:
            for dur in duration:
                if scenario in scenario_dfs and s in scenario_dfs[scenario] and dur in scenario_dfs[scenario][s]:
                    df = scenario_dfs[scenario][s][dur]
                    scenario_dfs[scenario][s][dur]["Minimum"] = df[df.columns[2:]].min(axis=1)
                    scenario_dfs[scenario][s][dur]["Maximum"] = df[df.columns[2:]].max(axis=1)
                    scenario_dfs[scenario][s][dur]["Average"] = df[df.columns[2:]].mean(axis=1)

    # Create plots for each scenario
    for scenario in scenario_list:
        for s in season:
            for dur in duration:
                if scenario in scenario_dfs and s in scenario_dfs[scenario] and dur in scenario_dfs[scenario][s]:
                    df = scenario_dfs[scenario][s][dur]
                    plt.figure(figsize=(12, 6))
                    plt.title(f"Scenario: {scenario}, Season: {s}, Duration: {dur}", fontsize=16)

                    plt.plot(df["time"], df["controller.u_T_HeatGrid_FF_set"], label="Controller Temperature",
                             color="black")
                    plt.plot(df["time"], df["Minimum"], label=f"{dur} Minimum", color="grey")
                    plt.plot(df["time"], df["Maximum"], label=f"{dur} Maximum", color="grey")
                    plt.plot(df["time"], df["Average"], label=f"{dur} Average", color="grey", linestyle="--")
                    plt.fill_between(df["time"], df["Minimum"], df["Maximum"], color="lightgrey")

                    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                    plt.xlabel("Time")
                    plt.ylabel("Temperature (Celsius)")
                    plt.tight_layout()

                    if save_as_pdf:
                        pdf_pages.savefig()

                    plt.show()

    if save_as_pdf:
        pdf_pages.close()
