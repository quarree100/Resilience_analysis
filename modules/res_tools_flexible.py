#author: Thies, 28.03.2022

import pandas as pd
import numpy as np
import itertools
import os
from modules.plotting_results import radar_chart

# The first functions are specific to the folder structure in the project.
# They read the dimension, weights, types of energy systems and the list of
# installed systems found in /excel_data/Diversity Index weighted.xlsx

def read_dimensions_and_weights(path_to_excel_file, sheet):
    id_sheet = pd.read_excel(path_to_excel_file, sheet_name=sheet, usecols="B:G", engine="openpyxl")   
    weight_dict = {}
    for dimension in list(id_sheet):
        weight_dict[dimension] = id_sheet[dimension][0]
    id_sheet = pd.DataFrame(id_sheet.iloc[1:, :])
    return [id_sheet, weight_dict]


def read_energy_system_types(path_to_excel_file = "excel_data/Diversity Index weighted.xlsx",sheet_name = "Anlagentypen"):

    path_to_excel_file = path_to_excel_file

    df = pd.read_excel(path_to_excel_file, sheet_name="Anlagentypen")
    df = df.set_index("Name")

    return df


def read_energy_systems(path_to_excel_file = "excel_data/Diversity Index weighted.xlsx",sheet_name = "Anlagen"):
    path_to_excel_file = path_to_excel_file
    
    df = pd.read_excel(path_to_excel_file, sheet_name=sheet_name, engine="openpyxl")
    df = df.set_index(["Index", "Brennstoff"], drop = False)

    return df


class EnergySystem:
    """energy systems, called via the index and fuel_type.
    """



    def __init__(self, system_index, fuel_type, excel_file = "Diversity Index weighted.xlsx",
                 sheet_name = "Anlagen", csv_file="Anlagentypen.CSV"):
        self.df_attr = pd.read_csv(csv_file, delimiter=";", encoding="latin", index_col="Name", skiprows=[1])
        self.df_syst = read_energy_systems(path_to_excel_file=excel_file, sheet_name=sheet_name)

        self.system_index = system_index #number of the energy system
        self.fuel_type = fuel_type #type of fuel, string value

        self.system_type = self.df_syst.loc[(self.system_index, self.fuel_type)]["Anlagentyp"] #type of system
        self.type_and_fuel = self.system_type+"_"+self.fuel_type #type and fuel connected with an "_" form the index for the stirling index calculations
        self.attributes = self.df_attr.loc[self.type_and_fuel] #a pd.Series of attribute names and their (string) values
        self.inout = self.df_syst.loc[(self.system_index,self.fuel_type)]["p_fuel":"p_el"]  #a pd.Series of energy flows and their (numeric) values


# the following two functions serve as translation from the excel sheet to a dataframe that the calculating functions can work with.



def summon_systems(path_to_excel_file = "excel_data/Diversity Index weighted.xlsx",
                   sheet_name = "Anlagen",
                   csv_file = "Anlagentypen.CSV"):
    #getting the data
    df = pd.read_excel(path_to_excel_file, sheet_name=sheet_name, engine="openpyxl")

    l_o_s = []
    for i in range(len(df)):
        info = df.iloc[i]
        new_system = EnergySystem(system_index = info["Index"],
                                    fuel_type = info["Brennstoff"],
                                  excel_file = path_to_excel_file,
                                  sheet_name = sheet_name,
                                  csv_file = csv_file
                                  )

        l_o_s.append(new_system)
    return l_o_s


# as it is now, functions below all use a list_of_systems that is the return value of summon_systems()


def unpacking_systems(list_of_systems, attributes = False, inout = False, metadata_on_attributes = False):
    '''
    this takes a list of EnergySystem objects and turns them into a pandas dataframe that can be used to calculate other things.
    The function will always parse the systems' indices, plus all of the parameters that have the value True.
    :param list_of_systems: list of EnergySystem objects. Often this will be called with summon_systems().
    :param fuel_type, system_type, attributes, inout: booleans for the different kinds of information stored in the EnergySystem object.
    :return: a pandas dataframe
    '''
    #columns:
    columns = ["system_index", "type_and_fuel"]
    if attributes:
        columns += [c for c in list_of_systems[0].attributes.keys()]
    if inout:
        columns += [k for k in list_of_systems[0].inout.keys()]

    #values
    values = []
    for s in list_of_systems:
        new_entry = [s.system_index, s.type_and_fuel]
        if attributes:
            new_entry += [a for a in s.attributes]
        if inout:
            new_entry += [e for e in s.inout]
        values.append(new_entry)

    df = pd.DataFrame(values, columns=columns)

    #metadata and return statement
    if metadata_on_attributes:
        mdat_att = list(list_of_systems[0].attributes.keys())
        return df, mdat_att
    else:
        return df


def shannon_index(list_of_systems, energy_provision):
    """The shannon index is based on the number of different types of energy sources and their relative energy shares.

    The function sums up the energy provision per unique id and calculates the shannon index based on
    Shannon, 1948.
    """
    new_df = unpacking_systems(list_of_systems,  inout = True)

    df_pivot = pd.pivot_table(
        new_df, index='type_and_fuel',
        values=[energy_provision],
        aggfunc={energy_provision: np.sum})
    total_energy = np.sum(df_pivot[energy_provision])
    df_pivot["relative_contribution"] = df_pivot[energy_provision]/total_energy
    for ind in df_pivot.index:
        df_pivot.at[ind, 'shannon_index'] = \
            - df_pivot.loc[ind, 'relative_contribution'] * \
            np.log(df_pivot.loc[ind, 'relative_contribution'])

    si_result = df_pivot['shannon_index'].sum()
    return si_result


# stirling index
def handling_attributes(data, id_column, attribute_columns, energy_provision_column):

    #create a new dataframe
    df = pd.DataFrame()
    df["type_and_fuel"] = data[id_column]
    df["energy_provision"] = data[energy_provision_column]

    #Now its time for the attributes.

    for dimension in attribute_columns:
        df[dimension] = data[dimension]

    return df


def calculate_disparity(a, b, weights):
    if len(a) != len(b):
        raise Exception("attribute vectors need to be same length")
    difference_list = []
    for i in range(len(a)):
        if a[i] == b[i]:
            dif = 0
        else:
            dif = 1
        difference_list.append(dif)
    # the coefficient takes the weighting of dimensions into account:
    difference_coefficient = np.array(weights).dot(np.array(difference_list)) / np.sum(weights)
    return difference_coefficient


def stirling_index(list_of_systems, energy_provision, alpha=1, beta=1,
                   filepath_for_dim_weights = "examples/excel_data/Diversity Index weighted.xlsx"):
    """

    :param list_of_systems: a list of EnergySystem objects.
    :param energy_provision: A string value, could be "p_inst_out_th", "p_inst_out_el", e.t.c., i.e., the power or energy who's
                                stirling-index shall be calculated
    param alpha and param beta:
        Weighting factors for different aspects of diversity.
                 alpha and beta may take values between 0 and 1. The values determine what the function is testing for:
                 alpha      beta        equivalent      aspect of biodiversity
                 0          0           (N^2-N)/2       variety(number of different entities)
                 0          1           (Gini)/2        balance and variety
                 1          0                           disparity and variety
                 1          1                           balance, disparity and variety

    :return: a numerical value for the stirling index.
    """
    #getting the weight of the dimensions as a list
    df = pd.read_csv(filepath_for_dim_weights, delimiter=";", encoding="latin", index_col="Name")
    dim_weight = df.loc["Gewichtung"]
    dim_weight = list(map(int, dim_weight.values))
    dim_weight = np.array(dim_weight)

    #parsing the energy systems into a df, creating a list of names of attributes (dimensions)
    df, attr_list = unpacking_systems(list_of_systems, attributes=True,inout=True,
                                      metadata_on_attributes=True)
    ###the dataframe now contains an "index" column that isn't needed, a type_and_fuel column acacting as an index,
    # attributes and many energy columns

    #leaving behind non-interesting energy and power columns
    wanted_columns = ["type_and_fuel"]+attr_list+[energy_provision]
    df = df.loc[:, wanted_columns]

    # getting some more information on the data
    no_of_dimensions = len(attr_list)
    total_energy_provided = np.sum(df[energy_provision])

    ###The next step is aggregating all the entities with the same type_and_fuel.
    # Beyond this point, the type_and_fuel column is no longer needed, but here,
    #it is used as an index. This is because, tehretically, there might be
    # two systems that have a different type but are the same in all the attributes.
    # whether this makes a mathematical difference I don't know.
    df_pivot = pd.pivot_table(data = df,
                              index = ["type_and_fuel"]+attr_list,
                              values = energy_provision,
                              aggfunc={energy_provision:np.sum})
    df_pivot[energy_provision] = pd.to_numeric(df_pivot[energy_provision])

    #convert absolute to relative energy provision
    df_pivot[energy_provision]/= total_energy_provided

    #calculating disparity and the index
    all_combinations = list(itertools.combinations(list(df_pivot.index), 2)) #list of all pairwise combinations of systems
    final_index = 0 #stirling index

    for combination in all_combinations:
        #calculating the disparity
        #calculating the respective relative contribution
        element1= combination[0]
        element2= combination[1]
        energy_1_rel = float(df_pivot.loc[[element1]].values)
        energy_2_rel = float(df_pivot.loc[[element2]].values)
        difference_coefficient = calculate_disparity(element1[1:], element2[1:], dim_weight) #the [1:] is bc the element1[0] is the type_and_fuel column, which should not be used for disparity.
        ###applying the stirling formula
        stirling_combination = difference_coefficient ** alpha * (energy_1_rel * energy_2_rel) ** beta
        final_index += stirling_combination
    return final_index


def redundancy(load, list_of_systems, value_column, alpha=0.1, beta=0.5):
    """

    :param load: a numeric value the required energy
    :param list_of_systems: a lists of objects of the class energy_system
    :param value_column: the name of the column for which the redundancy should be measured.
    :param alpha: a weighting factor for the importance of excess energy.
                  a value of 0 means that excess energy will be ignored.
    :param beta: A weighting factor for the importance of equality
    NOTE: With alpha = 0.1 and beta = 0.5, the two components of the redundancy are nearly equally rated
    :return:
    """
    #aggregate systems by index (for example if a system switches energy sources during the year)
    df = unpacking_systems(list_of_systems, inout = True)
    df_pivot = pd.pivot_table(data = df,
                              index = "system_index",
                              values = value_column,
                              aggfunc=np.max)
    energy_provision = sum(df_pivot[value_column])
    excess_energy = 1-load/energy_provision
    gini = 1-sum([(energy/energy_provision)**2 for energy in df_pivot[value_column]])
    res = gini**beta*excess_energy**alpha
    return res

def anlagen_table_convertor(store_results, scenarios=["A", "B", "C"],
                            anlagentypen=["air_heat_pump", "air_heat_pump", "electrolyzer", "BHKW (CHP)", "boiler"],
                            brennstoff=["9: Netzstrom", "9: Netzstrom", "9: Netzstrom", "1: Gas", "1: Gas"],
                            ETA_ELECTROLYSER=0.75, ETA_BOILER=0.9, index=["0", "1", "2", "3", "4"],
                            p_th_heat_pumps=500,
                            parameter_file="Parameter_Values.csv"):
    """ Takes the parameter values file for each scenario and generates a table with the Anlagentypen and their
    info, saving it as an excel file.

    Arguments:
        store_results: string. Path where the resulting table should be saved.
        scenarios: list of strings. Names of the scenarios considered in the input paremeter file. Default:
        ["A", "B", "C"].
        anlagentypen: list of strings. Names of the Anlagentypen considered in the input paremeter file. Default:
        ["air_heat_pump", "air_heat_pump", "electrolyzer", "BHKW (CHP)", "boiler"].
        brennstoff: list of strings. Names of the fuel types considered in the input paremeter file. Default:
        ["9: Netzstrom", "9: Netzstrom", "9: Netzstrom", "1: Gas", "1: Gas"].
        ETA_ELECTROLYSER: float. Efficiency of the electrolyser. Default: 0.75.
        ETA_BOILER: float. Efficiency of the boiler. Default: 0.9.
        index: list of strings. Indices of the Anlagen considered in the input parameter file. Default:
        ["0", "1", "2", "3", "4"].
        p_th_heat_pumps: integer. Thermal power of the heat pumps. Default: 500.
        parameter_file: string. Name of the input parameter file. Default: "Parameter_Values.csv".

    """

    # reading input file
    path = os.path.join("input", "common", "dimension_scenarios")
    df = pd.read_csv(os.path.join(path, parameter_file), delimiter=";")

    # creating the table to fill up with the input information
    table_info = {
        "Index": index,
        "Anlagentyp": anlagentypen,
        "Brennstoff": brennstoff,
        "p_fuel": np.zeros(len(index)),
        "p_th": np.zeros(len(index)),
        "p_el": np.zeros(len(index))}

    table = pd.DataFrame(table_info)

    # the output excel file is created
    writer = pd.ExcelWriter(os.path.join(store_results, "data", 'scenarios_system_dimensions.xlsx'))

    # for each scenario, all the missing power values are filled out according to the input parameters
    # and saved in a separate sheet in the excel file
    for scenario in scenarios:
        # electric power of chp
        chp_cap_el = float(df[f"{scenario}"].loc[df["Parameter"] == "capP_el_chp"])
        table.loc[table["Anlagentyp"] == "BHKW (CHP)", ["p_el"]] = chp_cap_el

        # fuel required and thermic power of chp
        chp_eta_el = float(df[f"{scenario}"].loc[df["Parameter"] == "eta_el_chp"])
        chp_eta_th = float(df[f"{scenario}"].loc[df["Parameter"] == "eta_th_chp"])
        p_fuel = chp_cap_el / chp_eta_el
        table.loc[table["Anlagentyp"] == "BHKW (CHP)", ["p_fuel"]] = - p_fuel
        table.loc[table["Anlagentyp"] == "BHKW (CHP)", ["p_th"]] = p_fuel * chp_eta_th

        # electric power of electrolyzer
        electrolyser_cap_el = float(df[f"{scenario}"].loc[df["Parameter"] == "capP_el_electrolyser"])
        electrolyser_gas = electrolyser_cap_el * ETA_ELECTROLYSER
        table.loc[table["Anlagentyp"] == "electrolyzer", ["p_el"]] = - electrolyser_cap_el
        table.loc[table["Anlagentyp"] == "electrolyzer", ["p_th"]] = electrolyser_cap_el - electrolyser_gas
        table.loc[table["Anlagentyp"] == "electrolyzer", ["p_fuel"]] = electrolyser_gas

        # thermic power of boiler
        boiler_cap_th = float(df[f"{scenario}"].loc[df["Parameter"] == "capQ_th_boiler"])
        table.loc[table["Anlagentyp"] == "boiler", ["p_th"]] = boiler_cap_th
        table.loc[table["Anlagentyp"] == "boiler", ["p_fuel"]] = - boiler_cap_th / ETA_BOILER

        # heat pumps
        table.loc[table["Anlagentyp"] == "air_heat_pump", ["p_th"]] = p_th_heat_pumps
        hp1_eta = float(df[f"{scenario}"].loc[df["Parameter"] == "ScaleFactor_HP1"])
        hp2_eta = float(df[f"{scenario}"].loc[df["Parameter"] == "ScaleFactor_HP2"])
        table.loc[table["Anlagentyp"] == "air_heat_pump", ["p_el"]] = [p_th_heat_pumps * hp1_eta,
                                                                       p_th_heat_pumps * hp2_eta]
        # the info of each scenario is saved in a separate sheet
        table.to_excel(writer, f'{scenario}', index=True)
        writer.save()


def resilience_attributes_calculation(store_results, excel_file='scenarios_system_dimensions.xlsx',
                                      csv_file="Anlagentypen.CSV", show_plot=True):
    """
    Calculates the Shannon index, the Stirling index and the redundancy of the system from the diversity information
    and the Anlagentypen table. If show_plot=True, it also creates a radar chart with the results for each scenario
    considered.

    Arguments:
        store_results: string. Path where the output will be saved.
        excel_file: string. Name of the excel file with the diversity information of the system. Default:
        'scenarios_system_dimensions.xlsx'.
        csv_file: string. Name of the csv file with the Anlagentypen table. Default: "Anlagentypen.csv".
        show_plot: boolean. Whether the plot should be made or saved.

    Returns:
        (if show_plot=False) attributes: list of np.arrays where these arrays correspond to the shannon and stirling
        indices and redundancy values for each scenarios.
    """

    # path of the input files are created and excel file is read
    path = os.path.join("input", "common", "dimension_scenarios")
    excel_file_path = os.path.join(store_results, "data", excel_file)
    xl = pd.read_excel(excel_file_path, sheet_name=None, engine="openpyxl")
    csv_file_path = os.path.join(path, csv_file)

    load = 290  # (aprox.) from excel file "Quarree100_load_15_Modelica"

    # empty lists for each index to be filled up with the corresponding values for each scenario
    redundancy_list = []
    stirling_list = []
    shannon_list = []

    # for each sheet in the excel file, corresponding to each scenario, the index values are obtained
    for sheet in xl.keys():
        l_o_s = summon_systems(path_to_excel_file=excel_file_path, sheet_name=sheet, csv_file=csv_file_path)
        redundancy_list.append(redundancy(load, l_o_s, "p_th"))
        shannon_list.append(shannon_index(l_o_s, "p_th"))
        stirling_list.append(stirling_index(l_o_s, "p_th", filepath_for_dim_weights=csv_file_path))

    # the lists are made into arrays for convenience
    redundancy_index = np.array(redundancy_list)
    shannon = np.array(shannon_list)
    stirling = np.array(stirling_list)

    #list with the results
    attributes = [shannon, stirling, redundancy_index]

    # Transformation of the variables to make the shapes in the plot more visible and different from each other
    # redundancy_transformed = np.zeros(4)
    # max_val = redundancy.max() + redundancy.mean()/5
    # min_val = redundancy.min() - redundancy.mean()/5
    # val_range = max_val - min_val
    # for count, element in enumerate(redundancy):
    #    redundancy_transformed[count] = (element - min_val) * 100 / val_range

    # stirling_transformed = np.zeros(4)
    # max_val = stirling.max() + stirling.mean()/5
    # min_val = stirling.min() - stirling.mean()/5
    # val_range = max_val - min_val
    # for count, element in enumerate(stirling):
    #    stirling_transformed[count] = (element - min_val) * 100 / val_range

    # shannon_transformed = np.zeros(4)
    # max_val = shannon.max() + shannon.mean()/5
    # min_val = shannon.min() - shannon.mean()/5
    # val_range = max_val - min_val
    # for count, element in enumerate(shannon):
    #    shannon_transformed[count] = (element - min_val) * 100 / val_range

    # The radar chart is done and saved in the "store_results" path
    if show_plot:
        scenarios = xl.keys()
        categories = ["Shannon Index", "Stirling Index", "Redundancy"]

        radar_chart(store_results=store_results, attributes=attributes,
                    scenarios=scenarios, categories=categories)


    # Saving attributes as a .csv file
    attributes_df = pd.DataFrame()
    attributes_df["shannon_index"] = attributes[0]
    attributes_df["stirling_index"] = attributes[1]
    attributes_df["redundancy"] = attributes[2]
    attributes_df.to_csv(os.path.join(store_results, "data", "resilience_attributes.csv"))

