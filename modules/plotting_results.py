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
