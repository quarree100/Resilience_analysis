import res_tools_flexible as res
import pandas as pd
import numpy as np

'''This script was written to go through function by function, to understand the functions provided in res_tools_flexible.py
'''

'''
"The main excel file"

The data stored on the different systems within the Modelica model can be found in the excel file
"excel_data/Diversity Index weighted.xlsx". I refer to this file as "the main excel file"
It contains three sheets:

sheet "Dimensionen":
 - all the dimensions by which the different energy systems are categorized and their possible values.
 - The row "Gewichtung" contains weights for the dimensions. These decide how important the dimension is for calculating 
   the diversity. values should be 0 or higher. 
   
sheet "Anlagentypen": 
 - a database for all the kinds of Systems that occur in the model.
 - The "Name" column, which serves as an index, is made up of the type of system+"_"+fuel source. fuel source ("Brennstoff")
   also has it's own column. In the future, maybe it would be nice to just make a compound index of the column 
   "Brennstoff" and "Name", and turn name into just the type of system.

sheet "Anlagen":
- This is where information on the actually installed systems is stored. The systems have an index ,
  a type of system, a fuel type and a lot of columns for inputs and outputs of energy in different forms. 
- There may be multiple rows with the same index but different fuel_types. 
  This is for the case in which a system switches fuel_type for a certain period of time.
- IMPORTANT: The index + fuel_type act as an index. Don't have multiple systems with the same index AND same fuel_type

'''

'''
The EnergySystem Class

This is the fundamental object that all the other functions work with. An EnergySystem object could, for example be 
a heat pump or a central heating plant. '''

#An EnergySystem Object needs to be called with it's index and it's fuel_type, like so:
my_energy_system = res.EnergySystem(0, "9: Netzstrom")

#it contains
#  - the input and output values found in the "Anlagen" sheet under the index and fuel type (main excel file1),
#  - the attributes stated in "Anlagentypen" for the systemtype and fuel that is specified in the "Anlagen" sheet
#  - some other individual values, but the things mentioned above are the most important.

#  -they can be called like so:

#energy in- and outputs:
inouts = my_energy_system.inout
print(inouts)
#returns a series

#attributes:
attrs = my_energy_system.attributes
print(attrs)
#also returns a series.

# This means that, for the VALUE a specific output, one could for example use:
my_energy_system.inout["p_inst_out_th"]
#or
my_energy_system.attributes["Aufstellungsort"]
#these can be ALTERED, like so:
my_energy_system.inout["p_inst_out_th"] # returns 500
my_energy_system.inout["p_inst_out_th"] = 200
# now
my_energy_system.inout["p_inst_out_th"] # returns 200.

# These energy systems are the basis for calculating the diversity and the redundancy of the system.

#per default, the EnergySystem init function will look for the index and fuel type in the excel sheet mentioned above.
# this can be changed with arguments however:

my_energy_system_2 = res.EnergySystem(system_index = 9, fuel_type="1: Gas")
#will result in an error bc there is no system with index 9 in our normal system.
# However, specifying another file and sheet helps:
my_energy_system_2 = res.EnergySystem(system_index = 9, fuel_type="1: Gas",
                                      path_to_excel_file = "excel_data/res_tools_example_data.xlsx",
                                      sheet_name ="Anlagen_more_indices")

#both the function "stirling_index" and "redundancy" take a list of systems as an input.
# To create this list of systems, there are a few utility functions.

list_of_systems = res.summon_systems()

#this reads all the systems in the "Anlagen" sheet of the main excel file and create a list of EnergySytem objects from it.
#One can change the sheet and file:
list_of_systems_2 = res.summon_systems(path_to_excel_file= "excel_data/res_tools_example_data.xlsx",
                                       sheet_name="Anlagen_more_systems_same_type")

#now that we have a list of systems to work with, we can calculate diversity and redundancy.

#For now, let's use the example data to calculate three metrics for resilience: Shannon index, Stirling Index, Redundancy:
example_path = "excel_data/res_tools_example_data.xlsx"
l_o_s_0 = res.summon_systems(path_to_excel_file=example_path,sheet_name="Anlagen_basic")

# the shannon index and the stirling index require two arguments: A list of systems,
# the name of the output that is being looked at.
shannon_0 = res.shannon_index(l_o_s_0, "p_inst_out_th")
stirling_0 = res.stirling_index(l_o_s_0, "p_inst_out_th")
# redundancy also requires the load of the system in addition to the other attributes.
redundancy_0 = res.redundancy(634, l_o_s_0, "p_inst_out_th")

print("stirling index: ", stirling_0,"\nshannon index:  ", shannon_0, "\nredundancy:     ", redundancy_0)

# these values are all relative. This means that they don't mean much on their own,
# but only in comparison to different configurations of systems.
# So let's do just that.

'''the stirling index under different conditions'''

# in the file excel_data/res_tools_example_data.xlsx, there are multiple versions of the "Anlagen" sheet.
# the sheet "Anlagen_examples_for_p_inst" contains mutliple combinations of values for installed power.
# The columns ,p_inst_out_th,p_inst_out_el,p_inst_out_H2 contain three different distributions of power.
# p_inst_out_th contains systems that all have equal power.
# p_inst_out_el also has equal distribution of power over all systems, but twice as much power per system
# and p_inst_out_H2  has one system that has more power than the others.
l_o_s_1 = res.summon_systems(path_to_excel_file=example_path,sheet_name="Anlagen_examples_for_p_inst")
for col in ["p_inst_out_th","p_inst_out_el","p_inst_out_H2"]:
    result = res.stirling_index(l_o_s_1,col)
    print (col, ":   ", result)

# As can be seen, there is no difference in diversity between the first two.
# This is because diversity is not a measure of
# how much power is there, but how it is distributed.
# A higher power with the same distribution will result in the same diverstiy.
# the third value is lower because one of the systems has more power, making the power less evenly distributed.

#if the number of systems changes, so does the stirling index. If we do the above example, but omit one of the systems, we get the following:

for begin_at in [0,1]:
    print ("beginning with the ",begin_at,"th element:")
    for col in ["p_inst_out_th","p_inst_out_el","p_inst_out_H2"]:
        result = res.stirling_index(l_o_s_1[begin_at:],col)
        print (col, ":   ", result)

# these results may be surprising. While it makes sense that the value for _H2 rises,
# because the element that had much higher power then the others was removed
# (and thus, the evenness of the powers was increased), it seems strange that the
# values for _th and _el also rise. The reason for this is that the stirling index aggregates systems with the
# same index and fuel type. Thus, the first two systems are treated as one system, with the powers added up.
# So, when removing the first system, the evenness of power distribution actually rises.

# If instead you remove the third system(which is unique), the stirling index WILL fall.

for i in range(2):
    print (["longer list:","shorter list:"][i])
    for col in ["p_inst_out_th","p_inst_out_el","p_inst_out_H2"]:
        result = res.stirling_index([l_o_s_1,l_o_s_1[:2]+l_o_s_1[3:]][i],col)
        print (col, ":   ", result)

#This means that for the stirling index, it doesn't matter how many systems there are, but how many DIFFERENT ones.
#This is why, even with double the amount of systems,
#the stirling index won't increase:

for no_of_reps in [1,2,3]:
    l_o_s = res.summon_systems()
    result = res.stirling_index(l_o_s*no_of_reps,"p_inst_out_th")
    print ("with ", len(l_o_s)*no_of_reps, " systems: ", result)

#finally, the stirling_index can also be influenced with the weighting of the dimensions, and the weighting of alpha,
# and beta. For the dimensions, you need to change the weighting of the dimensions in the excel file.
# For convenience, I've created an additional sheet with different weightings.

for sheet in ["Dimensionen","Dimensionen_diff_wgts"]:
    result = res.stirling_index(l_o_s_1,"p_inst_out_th",
                                filepath_for_dim_weights= "excel_data/res_tools_example_data.xlsx",
                                dim_weight_sheet= sheet)
    print ("with the dimension weightings from '", sheet, "':\n",result)


'''okay, now for the redundancy.'''

#redundancy needs the parameter "load". Here is how I got the load:

#the following function returns both the max load for electric as well as for thermal power
def get_max_load():
    filepath = "redundancy_sheets/Quarree100_load_15_Modelica.xlsx"
    df = pd.read_excel(filepath, header = 0)

    #setting the value of empty cells (there is just one) to 0
    names_of_columns = list(df)
    for column in names_of_columns:
        for i in range(len(df["E_el_GHD"])):
            if df[column][i] == ' ':
                print("alert:", column, i) #this is just to notify the user about the change
                df[column][i] = 0

    #summing up all the electric and thermic energy usages to find out the "installed" power for each of them

    df["sum_el"] = df["E_el_GHD"] + df["E_el_HH"]
    df["sum_th"] = df["E_th_RH_HH"] + df["E_th_TWE_HH"] + df["E_th_RH_GHD"] + df["E_th_TWE_GHD"] + df["E_th_KL_GHD"]

    max_load_el = max(df["sum_el"])
    max_load_th = max(df["sum_th"])
    return max_load_el,max_load_th


def get_thermic_load():
    res = get_max_load()[1]
    return res


load = get_thermic_load()

#from here, it's pretty stright forward: create a list of EnergySystem objects and see how well they manage!
#I figured out that you get into trouble if your exce sheet contains multiple occurneces of same index+same fueltype.
# don't do this! (or come up with  fix for it)

example_path = "excel_data/res_tools_example_data.xlsx"
l_o_s_2 = res.summon_systems(path_to_excel_file=example_path,sheet_name="Anlagen_examples_for_p_inst")

res.redundancy(load = load, list_of_systems=l_o_s_2,value_column="p_inst_out_th")

#lets calculate the redundancy for the different Anlagen_ - sheets:

for sheet in ["Anlagen_basic",
              "Anlagen_examples_for_p_inst",
              "Anlagen_more_indices",
              "Anlagen_more_indices_2"]:
    l_o_s = res.summon_systems(path_to_excel_file="excel_data/res_tools_example_data.xlsx",
                               sheet_name=sheet)
    result = res.redundancy(load,l_o_s,"p_inst_out_th")
    print ("for ",sheet,": ",result)

# now, let~s look at these values. Anlagen_basic has a lower redundancy score than Anlagen_examples_for_p_inst,
# even though the total power installed in Anlagen_basic is higher. This is because the power in Anlagen_examples
# is more evenly distributed.
# Similarly, Anlagen_more_indices has a better score than Anlagen_more_indices_2, bc in the latter one, the last two
# Systems are changed to be the same ID (but different fuel type) as the first two systems.
# Since redundancy aggregates (and creates the maximum) for system IDs,
# leaving the most dominant system (look at the values in the sheet) in place while reducing the number of systems, this
# reduces the evenness.

# Both of these effects get more clear if you decrease alpha, lets say to 0:

for sheet in ["Anlagen_basic",
              "Anlagen_examples_for_p_inst",
              "Anlagen_more_indices",
              "Anlagen_more_indices_2"]:
    l_o_s = res.summon_systems(path_to_excel_file="excel_data/res_tools_example_data.xlsx",
                               sheet_name=sheet)
    result = res.redundancy(load,l_o_s,"p_inst_out_th",alpha=0)
    print ("for ",sheet,": ",result)

#now, ONLY the evenness of distribution between system IDs matters.
#on the other hand, if we make alpha greater and beta smaller, things will look different:

for sheet in ["Anlagen_basic",
              "Anlagen_examples_for_p_inst",
              "Anlagen_more_indices",
              "Anlagen_more_indices_2"]:
    l_o_s = res.summon_systems(path_to_excel_file="excel_data/res_tools_example_data.xlsx",
                               sheet_name=sheet)
    result = res.redundancy(load,l_o_s,"p_inst_out_th",alpha=1,beta = 0)
    print ("for ",sheet,": ",result)