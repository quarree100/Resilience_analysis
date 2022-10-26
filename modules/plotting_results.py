import matplotlib.pyplot as plt
import pandas as pd
import parse

filenames = ["data/results_Scenario A_ErrorProfiles_input.csv",
             "data/results_Scenario A_ErrorProfiles_input_Boiler_14_10_18.csv",
             "data/results_Scenario A_ErrorProfiles_input_CHP_13_1_18.csv",
             "data/results_Scenario B_ErrorProfiles_input.csv",
             "data/results_Scenario B_ErrorProfiles_input_Boiler_14_10_18.csv",
             "data/results_Scenario B_ErrorProfiles_input_CHP_13_1_18.csv",
             "data/results_Scenario C_ErrorProfiles_input.csv",
             "data/results_Scenario C_ErrorProfiles_input_Boiler_14_10_18.csv",
             "data/results_Scenario C_ErrorProfiles_input_CHP_13_1_18.csv"]

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

def temperature_control():
    """"""
    data = pd.read_csv("data/results_Scenario A_ErrorProfiles_input.csv")
    data["fMU_PhyModel.temperature_HeatGrid_FF.T"] = data["fMU_PhyModel.temperature_HeatGrid_FF.T"] - 273.15
    temp_A = data[vars[8]]
    temp_A_set = data[vars[9]]
    data = pd.read_csv("data/results_Scenario A_ErrorProfiles_input_Boiler_14_10_18.csv")
    data["fMU_PhyModel.temperature_HeatGrid_FF.T"] = data["fMU_PhyModel.temperature_HeatGrid_FF.T"] - 273.15
    temp_A_error_boiler = data[vars[8]]
    data = pd.read_csv("data/results_Scenario A_ErrorProfiles_input_CHP_13_1_18.csv")
    data["fMU_PhyModel.temperature_HeatGrid_FF.T"] = data["fMU_PhyModel.temperature_HeatGrid_FF.T"] - 273.15
    temp_A_error_chp = data[vars[8]]

    data = pd.read_csv("data/results_Scenario B_ErrorProfiles_input.csv")
    data["fMU_PhyModel.temperature_HeatGrid_FF.T"] = data["fMU_PhyModel.temperature_HeatGrid_FF.T"] - 273.15
    temp_B = data[vars[8]]
    temp_B_set = data[vars[9]]
    data = pd.read_csv("data/results_Scenario B_ErrorProfiles_input_Boiler_14_10_18.csv")
    data["fMU_PhyModel.temperature_HeatGrid_FF.T"] = data["fMU_PhyModel.temperature_HeatGrid_FF.T"] - 273.15
    temp_B_error_boiler = data[vars[8]]
    data = pd.read_csv("data/results_Scenario B_ErrorProfiles_input_CHP_13_1_18.csv")
    data["fMU_PhyModel.temperature_HeatGrid_FF.T"] = data["fMU_PhyModel.temperature_HeatGrid_FF.T"] - 273.15
    temp_B_error_chp = data[vars[8]]

    data = pd.read_csv("data/results_Scenario C_ErrorProfiles_input.csv")
    data["fMU_PhyModel.temperature_HeatGrid_FF.T"] = data["fMU_PhyModel.temperature_HeatGrid_FF.T"] - 273.15
    temp_C = data[vars[8]]
    temp_C_set = data[vars[9]]
    data = pd.read_csv("data/results_Scenario C_ErrorProfiles_input_Boiler_14_10_18.csv")
    data["fMU_PhyModel.temperature_HeatGrid_FF.T"] = data["fMU_PhyModel.temperature_HeatGrid_FF.T"] - 273.15
    temp_C_error_boiler = data[vars[8]]
    data = pd.read_csv("data/results_Scenario C_ErrorProfiles_input_CHP_13_1_18.csv")
    data["fMU_PhyModel.temperature_HeatGrid_FF.T"] = data["fMU_PhyModel.temperature_HeatGrid_FF.T"] - 273.15
    temp_C_error_chp = data[vars[8]]

    fig, axs = plt.subplots(4, 1, figsize=(15, 8))

    axs[0].plot(data["time"], temp_A, "b-")
    axs[0].plot(data["time"], temp_A_error_boiler, "b--")
    axs[0].plot(data["time"], temp_A_error_chp, "b:")
    axs[0].plot(data["time"], temp_A_set, "r-")
    axs[0].set_title("Scenario A")
    axs[0].set_ylabel("Temperature (C)")

    axs[1].plot(data["time"], temp_B, "g-")
    axs[1].plot(data["time"], temp_B_error_boiler, "g--")
    axs[1].plot(data["time"], temp_B_error_chp, "g:")
    axs[1].plot(data["time"], temp_B_set, "r-")
    axs[1].set_title("Scenario B")
    axs[1].set_ylabel("Temperature (C)")

    axs[2].plot(data["time"], temp_C, "c-")
    axs[2].plot(data["time"], temp_C_error_boiler, "c--")
    axs[2].plot(data["time"], temp_C_error_chp, "c:")
    axs[2].plot(data["time"], temp_C_set, "r-")
    axs[2].set_title("Scenario C")
    axs[2].set_ylabel("Temperature (C)")

    axs[3].plot(data["time"], temp_A, "b-", label="T without error, scenario A")
    axs[3].plot(data["time"], temp_A_error_boiler, "b--", label="T with boiler error, scenario A")
    axs[3].plot(data["time"], temp_A_error_chp, "b:", label="T with CHP error, scenario A")
    axs[3].plot(data["time"], temp_B, "g-", label="T without error, scenario B")
    axs[3].plot(data["time"], temp_B_error_boiler, "g--", label="T with boiler error, scenario B")
    axs[3].plot(data["time"], temp_B_error_chp, "g:", label="T with CHP error, scenario B")
    axs[3].plot(data["time"], temp_C, "c-", label="T without error, scenario C")
    axs[3].plot(data["time"], temp_C_error_boiler, "c--", label="T with boiler error, scenario C")
    axs[3].plot(data["time"], temp_C_error_chp, "c:", label="T with CHP error, scenario C")
    axs[3].plot(data["time"], temp_C_set, "r-", label="set T")
    axs[3].set_title("All scenarios")
    axs[3].set_ylabel("Temperature (C)")
    axs[3].set_xlabel("Time (min)")
    fig.legend()

    fig.tight_layout()
    plt.savefig("temperature_control.png")


def resilience_box_plot(data_file="data/resilience.csv", scenarios=["A", "B", "C"],
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

    Scenario_A = RI_df["A"]
    Scenario_B = RI_df["B"]
    Scenario_C = RI_df["C"]

    columns = [Scenario_A, Scenario_B, Scenario_C]
    fig, ax = plt.subplots()
    ax.boxplot(columns)
    plt.xticks([1, 2, 3], ["Scenario A", "Scenario B", "Scenario C"], rotation=10)
    plt.ylabel("Resilience Index")
    plt.title("Resilience Index")
    plt.tight_layout()
    plt.savefig("resilience_boxplot.png")

    # And we also save the new data frame with averages as well
    RI_df.loc[len(RI_df.index)] = [RI_df["A"].mean(), RI_df["B"].mean(), RI_df["C"].mean()]
    RI_df.index = ["No-error", "Boiler_14_10_18", "CHP_13_1_18", "Average"]
    RI_df.to_csv("Scenarios_resilience_w_average.csv")


def separate_plots(filename="data/results_Scenario A_ErrorProfiles_input.csv", vars=vars, scenarios=["A", "B", "C"]):
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

def plot(data_file="data/results_Scenario A_ErrorProfiles_input.csv",
         vars=vars, scenarios=["A", "B", "C"], show=True):
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
    fig, axs = plt.subplots(4, 2, figsize=(12, 12))

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

    axs[0, 0].plot(data["time"], data[vars[0]])
    axs[0, 0].set_title("Boiler Production", fontsize=14)

    axs[0, 1].plot(data["time"], data[vars[1]])
    axs[0, 1].set_title("CHP Production", fontsize=14)

    axs[1, 0].plot(data["time"], data[vars[2]])
    axs[1, 0].set_title("Electrolyzer Production", fontsize=14)

    axs[2, 0].plot(data["time"], data[vars[3]], label="Heat Pump 1")
    axs[2, 0].plot(data["time"], data[vars[4]], label="Heat Pump 2")
    axs[2, 0].set_title("Heatpumps Production", fontsize=14)
    axs[2, 0].legend()

    axs[1, 1].plot(data["time"], data[vars[5]])
    axs[1, 1].set_title("Qdot Production", fontsize=14)

    axs[2, 1].plot(data["time"], data[vars[6]])
    axs[2, 1].plot(data["time"], data[vars[7]], linestyle="-")
    axs[2, 1].set_title("Heatload", fontsize=14)

    axs[3, 0].plot(data["time"], data[vars[8]], label="T")
    axs[3, 0].plot(data["time"], data[vars[9]], label="set T")
    axs[3, 0].legend()
    axs[3, 0].set_title("Temperature", fontsize=14)

    fig.tight_layout()

    plt.show()

    # plt.savefig("data/" + title.replace(" ", "_") + ".png")


if __name__ == '__main__':
    for file in filenames:
        plot(data_file=file, show=True)
