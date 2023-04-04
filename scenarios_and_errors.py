import pandas as pd
import os


def read_scenarios_names(input_file="Parameter_Values.csv"):

    input_path = os.path.join("input", "common", "dimension_scenarios", input_file)

    df = pd.read_csv(input_path, delimiter=";")

    scenarios = []
    for scenario in df.columns[3:]:
        scenarios.append(scenario)

    return scenarios
