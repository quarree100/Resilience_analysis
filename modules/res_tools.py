# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import itertools
from matplotlib import pyplot as plt
# import logging


def remove_duplicates(values):
    output = []
    seen = set()
    for value in values:
        # If value has not been encountered yet,
        # ... add it to both list and set.
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def redundancy_index(data_i, demand, n_x):

    # exemplarisch berechnung
    # take any element of dict data
    # data_example = next(iter(data.values()))

    # ID of technologies, not considered for redundancy
    l_drop_technology = [4, 8, 9, 101]  # 4: Saisonalspeicher, 8: Solarthermie, 9: PV, 101: Wärmeübergabestation,
    l_drop_variant = [3]    # Oberfflächenwasser als Wärmequelle

    # just define separate dataframe
    df = data_i

    df_demand = pd.DataFrame()
    df_demand['demand'] = demand['Wärmeverbrauch'] + demand['Netz-/Speicherverluste']

    # energy_demand_total = df_demand['demand'].sum()

    # drop elements, which are not considered for redundancy
    for m in l_drop_technology:
        df = df[df['technology'] != m]

    for v in l_drop_variant:
        df = df[df['heat source (hp)'] != v]

    # get number of heat generators
    num = len(df.index)

    # add all installed power
    # installed_power = df.iloc[:, 3].sum()

    # add probability to dataframe
    df['prop'] = 0.03

    # generate list with tuples for each generation plant
    l_gen_i = []
    for i in range(num):
        l_gen_i += [(1, 0)]

    # logging.info('Calculate ALL combinations')

    risk_total = 0
    risk_total_nn = 0
    p_total = 0
    risk_s = np.zeros(num + 1)
    risks_i = [[]] * (len(l_gen_i) + 1)
    d_risks = {}
    for n in range(len(l_gen_i) + 1):
        d_risks.update({n: []})

    if n_x == 'all':
        # generate all combinations of on/off
        kombis = list(itertools.product(*l_gen_i))

        for k in kombis:
            # make an arry out of tuple
            status = np.array(k)

            # calculate remaining power
            # 0 := Ausfall;
            # 1 := Anlage fällt nicht aus;
            power = np.sum(df.loc[:, 'power'].values * status)
            df_demand['power'] = power
            df_demand['energy_lost'] = df_demand['demand'] - power

            # Energie which cannot be delivered
            energy_lost = df_demand.loc[df_demand['demand'] > power,
                                        'energy_lost'].sum()

            # probabilities of events of each generation plant
            prop = 1 - np.absolute(1 - status - df['prop'])
            prop_total = np.prod(prop)

            risk_i = prop_total*energy_lost

            risk_total += risk_i
            p_total += prop_total

            risk_s[num - sum(k)] += risk_i
            d_risks[num - sum(k)].append(risk_i)

    else:
        for i in range(n_x):
            print(i)

            # logging.info('Start Calculate next kombis_i')

            # variant A
            # kombis_i = [x for x in kombis if np.sum(x) == num - i]

            # # variant B
            # a = np.ones(num)
            # a[:i] = 0
            # kombis_i = list(itertools.permutations(a))
            # kombis_i = remove_duplicates(kombis_i)

            # variant C

            ind = np.arange(num)
            kombi_ind = list(itertools.combinations(ind, i))

            kombis_i = []
            for k in kombi_ind:
                a = np.ones(num)
                for l in k:
                    a[l] = 0
                kombis_i.append(a)

            # logging.info('Finish calculation kombis_i')

            for k in kombis_i:

                # make an arry out of tuple
                status = np.array(k)

                # calculate remaining power
                # 0 := Ausfall;
                # 1 := Anlage fällt nicht aus;
                power = np.sum(df.loc[:, 'power'].values * status)
                df_demand['power'] = power
                df_demand['energy_lost'] = df_demand['demand'] - power

                # Energie which cannot be delivered
                energy_lost = df_demand.loc[df_demand['demand'] > power,
                                            'energy_lost'].sum()

                # probabilities of events of each generation plant
                prop = 1 - np.absolute(1 - status - df['prop'])
                prop_total = np.prod(prop)

                risk_i = prop_total * energy_lost

                risk_i_nn = prop_total * energy_lost / \
                    (df_demand['demand'].sum() * len(kombis_i))

                risk_total += risk_i        # absolute risk value in [kWh]
                risk_total_nn += risk_i_nn  # relativ value normed with binominal koefficient and Energy demand

                p_total += prop_total

                risk_s[int(num - sum(k))] += risk_i
                d_risks[int(num - sum(k))].append(risk_i)

            # logging.info('Finish calculation RISK for kombis_i')
            print('Kombi i finished.')

    # optional: norm with total energy demand
    risk_total_n = 1 - (risk_total / df_demand['demand'].sum())

    risk_total_nnn = 1 - (risk_total_nn / n_x)

    risk_detail_df = pd.DataFrame(index=list(d_risks.keys()))

    if n_x != 'all':
        for i in range(n_x):
            risk_detail_df.at[i, 'num'] = len(d_risks[i])
            arr = np.asarray(d_risks[i])
            risk_detail_df.at[i, 'sum'] = arr.sum()
            risk_detail_df.at[i, 'mean'] = arr.mean()
            risk_detail_df.at[i, 'median'] = np.median(arr)
            risk_detail_df.at[i, 'min'] = arr.min()
            risk_detail_df.at[i, 'max'] = arr.max()
    else:
        for i in range(num+1):
            risk_detail_df.at[i, 'num'] = len(d_risks[i])
            arr = np.asarray(d_risks[i])
            risk_detail_df.at[i, 'sum'] = arr.sum()
            risk_detail_df.at[i, 'mean'] = arr.mean()
            risk_detail_df.at[i, 'median'] = np.median(arr)
            risk_detail_df.at[i, 'min'] = arr.min()
            risk_detail_df.at[i, 'max'] = arr.max()

    results_redundancy_detail = {'risk_total': risk_total,
                                 'risk_total_normed': risk_total_n,
                                 'risk_total_normed_nnn': risk_total_nnn,
                                 'risk_detail': d_risks,
                                 'risk_detail_df': risk_detail_df}

    return results_redundancy_detail


def reset_ids_kp(data_all):

    for key in list(data_all.keys()):

        data_i = data_all[key]

        df = data_i['generation']

        # ID of technologies, not considered for redundancy
        l_drop_generator = [3, 4, 9, 101]

        # drop elements, which are not considered for redundancy
        for n in l_drop_generator:
            df = df[df['ID Technologie'] != n]

        # insert new column with namedtuple of technology
        # Anlagenzuordnung

        for ind, row in df.iterrows():
            df.at[ind, 'power'] = row['Thermische Leistung [kW_th] / ' +
                                      'Solarfläche [m²] / Netzlänge [m] /' +
                                      ' Netz- bzw. Speichervolumen [m³]']
            # Brennwertkessel
            if row['ID Technologie'] == 1:
                df.at[ind, 'technology'] = int(1)
                df.at[ind, 'type'] = 1
            # Niedertemperaturkessel
            if row['ID Technologie'] == 2:
                df.at[ind, 'technology'] = 1
                df.at[ind, 'type'] = 2
            # KWK
            if row['ID Technologie'] == 6:
                df.at[ind, 'technology'] = 2
                df.at[ind, 'type'] = 3
            # Erdgas
            if row['ID Brennstoff'] == 1:
                df.at[ind, 'commodity'] = 1
                df.at[ind, 'specification'] = 1
            # Wärmepumpe
            if row['ID Technologie'] == 7:
                df.at[ind, 'technology'] = 3
                df.at[ind, 'type'] = row['ID Wärmepumpenwärmequelle'] + 3
                df.at[ind, 'commodity'] = 3
                if row['Verbrauchter Strom [kWh_el]'] - row['Fremdbezogener Strom bei Eigenstromnutzung [kWh_el]'] > 10:
                    df.at[ind, 'specification'] = 7
                else:
                    df.at[ind, 'specification'] = 6
            if row['ID Technologie'] == 8:
                df.at[ind, 'technology'] = 4
                df.at[ind, 'type'] = 8
                df.at[ind, 'commodity'] = 4
                df.at[ind, 'specification'] = 7

            # energy
            if row['ID Technologie'] != 7:
                df.at[ind, 'energy'] = row['Erbrachte Wärmeenergie [kWh_th]']
            else:

                if row['Verbrauchter Strom [kWh_el]'] - row['Fremdbezogener Strom bei Eigenstromnutzung [kWh_el]'] > 10:
                    share_net = row['Fremdbezogener Strom bei Eigenstromnutzung [kWh_el]'] / row['Verbrauchter Strom [kWh_el]']
                    share_eigen = 1 - share_net

                    df.at[ind, 'energy'] = share_eigen * row['Erbrachte Wärmeenergie [kWh_th]']

                    df.loc[ind+100] = df.loc[ind]
                    df.at[ind+100, 'power'] = 0   # set power to 0, because it is just a virtual plant for diversity calculations
                    df.at[ind+100, 'specification'] = 6
                    df.at[ind+100, 'energy'] = share_net * row['Erbrachte Wärmeenergie [kWh_th]']

                else:
                    df.at[ind, 'energy'] = row['Erbrachte Wärmeenergie [kWh_th]']

        # Change Datatype of new columns to integer
        try:
            df = df.astype({'technology': 'int32',
                            'type': 'int32',
                            'commodity': 'int32',
                            'specification': 'int32'})
        except ValueError:
            print("There are missing IDs for 'technology', 'type', 'commodity' " +
                  "or 'specification' of the generators.")

        df['ioew_id'] = \
            df['technology'].astype(str) + df['type'].astype(str) + \
            df['commodity'].astype(str) + df['specification'].astype(str)

        energy_total_1 = df['energy'].sum()
        energy_total_2 = data_i['generation'][
            'Erbrachte Wärmeenergie [kWh_th]'].sum()

        if round(energy_total_1, 0) != round(energy_total_2, 0):
            print(key, ' : Energiebilanz stimmt nicht überein.' +
                  ' Wärmeerzeuger vergessen!')

        df['energy_share'] = df['energy'] / energy_total_1

        # add additional id specific for each generator
        u_ids = list(df['ioew_id'].drop_duplicates().values)

        for uid in u_ids:
            indices = df.index[df['ioew_id'] == uid].tolist()
            i = 1
            for ind in indices:
                df.at[ind, 'ioew_id_2'] = uid + '-' + str(i)
                i += 1

        data_all[key].update({'heat_generation': df})

    return data_all


def calc_ev_shares(df, key_column):

    l_add = []  # list for additional rows

    if key_column == 'Eigenverbrauchter Strom aus der PV [kWh_el]':
        elec_source_id = 2
    elif key_column == 'Eigenverbrauchter Strom aus der KWK [kWh_el]':
        elec_source_id = 3
    else:
        elec_source_id = 999
        print('ACHTUNG: Fehlerhafte Zuordnung der Quelle des Eigenverbrauchten Stromes (KWK, PV ,.etc')

    for r, k in df.iterrows():
        # Falls Anlage Wärmepumpe ist
        if k['technology'] == 7 or k['technology'] == 5:
            # falls kein Eigenverbrauch Strom vorliegt
            if k[key_column] == 0:
                df.at[r, 'energy'] = k['Erbrachte Wärmeenergie [kWh_th]']
                df.at[r, 'electricity source'] = 1
            # falls Eigenverbrauch Strom vorliegt
            else:
                share_pv = k[key_column] / \
                           k['Verbrauchter Strom [kWh_el]']
                df.at[r, 'electricity source'] = 1
                df.at[r, 'energy'] = k['Erbrachte Wärmeenergie [kWh_th]'] * (
                            1 - share_pv)
                k_copy = k
                k_copy['electricity source'] = elec_source_id
                k_copy['power'] = 0
                k_copy['energy'] = k_copy[
                                       'Erbrachte Wärmeenergie [kWh_th]'] * share_pv
                l_add.append(k_copy)

        else:
            # Fälle, alles was nicht Heizstab oder Wärmepumpe ist
            df.at[r, 'electricity source'] = 0
            df.at[r, 'energy'] = df.loc[r, 'Erbrachte Wärmeenergie [kWh_th]']

    if l_add:
        df = df.append(l_add, ignore_index=True).copy()

    return df


def reset_ids_nk(data_all):

    for key in list(data_all.keys()):

        data_i = data_all[key]

        df = data_i['generation']
        # ID of technologies, not considered for redundancy
        l_drop_generator = [4, 9, 101]

        # drop elements, which are not considered for redundancy
        for n in l_drop_generator:
            df = df[df['ID Technologie'] != n]

        # drop useless columns
        l_drop_col = [
            'ID Wärmeverbund', 'Technologie', 'Brennstoff',
            'Stromflussrichtung', 'Saldo Speicher [kWh_th]',
            'Fehlendes Warmwasser [kWh_th]', 'Fehlende Heizung [kWh_th]',
            'Anteil PV', 'Gasmix', 'Volumen Speicher [m³]',
            'Anzahl der zusammengefassten Anlagen', 'Preisaufschlag?']

        df = df.drop(l_drop_col, axis=1)
        # insert new column with namedtuple of technology
        # Anlagenzuordnung

        data_all[key].update({'heat_generation': df})

        # create short dataframe
        df['ID'] = key.split('.')[0]
        df['Gebiet'] = key.split('UWB_')[1][:2]

        if df['Gebiet'].values[0] == 'NK':

            df['Verbrauchsszenario'] = key.split('-')[1].split('_')[0]

            if 'heute' in key:
                df['Betrachtungsjahr'] = 'heute'
            else:
                df['Betrachtungsjahr'] = key.split('_')[3]

            df['Erzeugerszenario'] = key.split('-')[-1].split('_')[0]

        elif df['Gebiet'].values[0] == 'KP':

            df['Betrachtungsgebiet'] = key.split('-')[1]
            df['Verbrauchsszenario'] = key.split('-')[2]
            df['Erzeugerszenario'] = key.split('-')[-1].split('_')[0]

        else:
            print('Gebiet ist weder KP noch NK!!!')

        df['Zeitstempel'] = key[-13:-4]

        l_columns = ['Strombezug', 'technology', 'fuel', 'heat source (hp)',
            'electricity source', 'power', 'energy']

        for c in l_columns:
            df[c] = np.nan

        df['power'] = df[
            'Thermische Leistung [kW_th] / Solarfläche [m²] / Trassenlänge [m] / Netz- bzw. Speichervolumen [m³]']
        df['technology'] = df['ID Technologie']
        df['fuel'] = df['ID Brennstoff']
        df['heat source (hp)'] = df['ID Wärmepumpenwärmequelle']

        dict_df = {'nur Stromnetz': df.copy(),
                   'PV Eigenverbrauch': df.copy(),
                   'KWK Eigenverbrauch': df.copy()}

        # Aufteilung der PV / KWK Strom für Wärmepumpe
        # Strom bezug nur aus dem Netz
        for s, df_i in dict_df.items():

            if s == 'nur Stromnetz':
                df_i['Strombezug'] = 'nur Stromnetz'
                df_i['energy'] = df_i['Erbrachte Wärmeenergie [kWh_th]']

                for r, k in df_i.iterrows():
                    if k['technology'] == 7 or k['technology'] == 5:
                        df_i.at[r, 'electricity source'] = 1
                    else:
                        df_i.at[r, 'electricity source'] = 0

            if s == 'PV Eigenverbrauch':
                df_i['Strombezug'] = 'PV Eigenverbrauch'

                key_col = 'Eigenverbrauchter Strom aus der PV [kWh_el]'

                df_i = calc_ev_shares(df_i, key_col)

            if s == 'KWK Eigenverbrauch':
                df_i['Strombezug'] = 'KWK Eigenverbrauch'

                key_col = 'Eigenverbrauchter Strom aus der KWK [kWh_el]'

                df_i = calc_ev_shares(df_i, key_col)

            df_i['elec_consumption'] = df_i['Verbrauchter Strom [kWh_el]']
            df_i['elec_pv_eigen'] = df_i['Eigenverbrauchter Strom aus der PV [kWh_el]']
            df_i['elec_kwk_eigen'] = df_i['Eigenverbrauchter Strom aus der KWK [kWh_el]']

            # Change Datatype of new columns to integer
            try:
                df_i = df_i.astype({'technology': 'int32',
                                  'fuel': 'int32',
                                  'heat source (hp)': 'int32',
                                  'electricity source': 'int32'})
            except ValueError:
                print(
                    "There are missing IDs for 'technology', 'type', " +
                    "'commodity', 'electricity source' or " +
                    "'specification' of the generators.")

            # check if sum of energies match?!
            energy_total_1 = df_i['energy'].sum()
            energy_total_2 = data_i['heat_generation'][
                'Erbrachte Wärmeenergie [kWh_th]'].sum()

            if (round(energy_total_1, 0) - round(energy_total_2, 0)) > 1:
                print(key, ' : Energiebilanz stimmt nicht überein.' +
                      ' Wärmeerzeuger vergessen!')
            else:
                print(key, 'Wärmemengen Test bestanden!')

            # calculate energy share of each generation unit
            df_i['energy_share'] = df_i['energy'] / energy_total_1

            # insert some columns and descriptions
            df_i['ioew_id'] = \
                df_i['technology'].astype(str) + df_i['fuel'].astype(str) + \
                df_i['heat source (hp)'].astype(str) + df_i['electricity source'].astype(str)

            lookup_tech = {1: 'Brennwert Kessel',
                           2: 'Niedertemperatur Kessel',
                           3: 'FW-Einspeisung',
                           5: 'Power-to-Heat',
                           6: 'KWK',
                           7: 'Wärmepumpe',
                           8: 'Solarthermie'}

            df_lookup_tech = pd.DataFrame.from_dict(lookup_tech,
                                                    orient='index',
                                                    columns=['Technologie'])

            lookup_fuel = {0: '',
                           1: 'Erdgas',
                           2: 'Biomethan',
                           3: 'Heizöl',
                           4: 'Kohle',
                           5: 'Biomasse'}

            df_lookup_fuel = pd.DataFrame.from_dict(lookup_fuel,
                                                    orient='index',
                                                    columns=['Technologie'])

            lookup_hp = {0: '',
                         1: 'Außenluft',
                         2: 'Untergrund (oberflächennah)',
                         3: 'Oberflächengewässer',
                         4: 'Abwasser',
                         5: 'KWK-NT-Abwärme',
                         6: 'Abwärme allgemein',
                         7: 'Untergrund (300 m)',
                         8: 'Untergrund (3 km)'}

            df_lookup_hp = pd.DataFrame.from_dict(lookup_hp, orient='index',
                                                  columns=['Technologie'])

            lookup_elec = {0: '',
                           1: 'Stromnetz',
                           2: 'PV Strom',
                           3: 'KWK Strom'}

            df_lookup_elec = pd.DataFrame.from_dict(lookup_elec, orient='index',
                                                  columns=['Technologie'])

            df_i['technology_n'] = df_i['technology'].apply(
                lambda x: df_lookup_tech.at[x, 'Technologie'])
            df_i['fuel_n'] = df_i['fuel'].apply(
                lambda x: df_lookup_fuel.at[x, 'Technologie'])
            df_i['heat source (hp)_n'] = df_i['heat source (hp)'].apply(
                lambda x: df_lookup_hp.at[x, 'Technologie'])
            df_i['electricity source_n'] = df_i['electricity source'].apply(
                lambda x: df_lookup_elec.at[x, 'Technologie'])

            df_i['Technologie'] = df_i['technology_n'] + ' ' + df_i['fuel_n'] + ' ' + df_i['heat source (hp)_n'] + ' ' + df_i['electricity source_n']

            # add additional id specific for each generator
            u_ids = list(df_i['ioew_id'].drop_duplicates().values)

            for uid in u_ids:
                indices = df_i.index[df_i['ioew_id'] == uid].tolist()
                i = 1
                for ind in indices:
                    df_i.at[ind, 'ioew_id_2'] = uid + '-' + str(i)
                    df_i.at[ind, 'Technologie und Strombezug'] = df_i.at[ind, 'technology_n'] + ' ' + df_i.at[ind, 'fuel_n'] + ' ' + df_i.at[ind, 'heat source (hp)_n']
                    df_i.at[ind, 'Anlage und Strombezug'] = df_i.at[ind, 'Technologie'] + ' - ' + str(i)
                    df_i.at[ind, 'Anlage'] = df_i.at[ind, 'technology_n'] + ' ' + df_i.at[ind, 'fuel_n'] + ' ' + df_i.at[ind, 'heat source (hp)_n'] + ' - ' + str(i)
                    i += 1

            # delete unused stuff
            df_i = df_i.drop(list(df_i.columns)[:21], axis=1)

            dict_df[s] = df_i

        data_all[key].update({'heat_generation_detail': dict_df})

    return data_all


def shannon_index(data_i):

    df_pivot = pd.pivot_table(
        data_i, index=['id_fgres'],
        values=['energy_share'],
        aggfunc={'energy_share': np.sum})

    for ind in df_pivot.index:
        df_pivot.at[ind, 'shannon_index'] = \
            - df_pivot.loc[ind, 'energy_share'] * \
            np.log(df_pivot.loc[ind, 'energy_share'])

    si = df_pivot['shannon_index'].sum()

    return si


def gini_index(data_i):

    df_pivot = pd.pivot_table(
        data_i, index=['id_fgres'],
        values=['energy_share'],
        aggfunc={'energy_share': np.sum})

    for ind in df_pivot.index:
        # formula for index
        df_pivot.at[ind, 'gini_index'] = \
            df_pivot.loc[ind, 'energy_share'] ** 2

    gini = 1 - df_pivot['gini_index'].sum()

    return gini


def lose_kopplung(data_i):

    df_pivot = pd.pivot_table(
        data_i, index=['Kopplung'],
        values=['energy_share'],
        aggfunc={'energy_share': np.sum})

    lk = 1 - df_pivot.at[0, 'energy_share']

    return lk


def eigenanteil(data_i):

    df_pivot = pd.pivot_table(
        data_i, index=['Eigenerzeugung'],
        values=['energy_share'],
        aggfunc={'energy_share': np.sum})

    ea = 1 - df_pivot.at[0, 'energy_share']

    return ea


def stirling_index(data_i):

    alpha = 1
    beta = 1

    l_div_dim = ['Ressourcenbasis', 'Infrastrukturabhängigkeit', 'Brennstoff',
                 'Technologieart', 'Technologieklasse']

    df_pivot = pd.pivot_table(
        data_i, index=['id_fgres'],
        values=l_div_dim+['energy_share'],
        aggfunc={'energy_share': np.sum,
                 'Ressourcenbasis': np.mean,
                 'Infrastrukturabhängigkeit': np.mean,
                 'Brennstoff': np.mean,
                 'Technologieklasse': np.mean,
                 'Technologieart': np.mean})

    # weighenting factor:
    g = pd.Series({'Ressourcenbasis': 1,
                   'Infrastrukturabhängigkeit': 1,
                   'Brennstoff': 1,
                   'Technologieklasse': 1,
                   'Technologieart': 1})

    p = list(itertools.combinations(list(df_pivot.index), 2))

    s = 0   # stirling index
    for i in p:

        # diversity distance d_i - old calculation
        # d_vector = np.compare_chararrays(list(i[0]), list(i[1]), "!=", True)

        d_1 = df_pivot.loc[i[0], l_div_dim]
        d_2 = df_pivot.loc[i[1], l_div_dim]

        d_delta = d_1 - d_2

        # generate 1/0 series
        d_binary = d_delta.apply(lambda x: 1 if x != 0 else 0)

        d_i = np.array(g).dot(np.array(d_binary)) / np.sum(g)

        # share p_i, p_j
        p_i = df_pivot.at[i[0], 'energy_share']
        p_j = df_pivot.at[i[1], 'energy_share']

        # contribution of combination i to stirling index
        s_i = d_i**alpha * (p_i*p_j)**beta

        s += s_i

    return s


def stirling_index_nk(data_i):

    alpha = 1
    beta = 1

    df_pivot = pd.pivot_table(
        data_i['heat_generation'], index=['ioew_id'],
        values=['technology', 'fuel', 'heat source (hp)',
                'energy_share'],
        aggfunc={'energy_share': np.sum,
                 'technology': np.mean,
                 'fuel': np.mean,
                 'heat source (hp)': np.mean})

    # weighenting factor:
    g = pd.Series({'technology': 1, 'fuel': 1, 'heat source (hp)': 1})

    p = list(itertools.combinations(list(df_pivot.index), 2))

    s = 0   # stirling index
    for i in p:

        # diversity distance d_i
        d_vector = np.compare_chararrays(list(i[0]), list(i[1]), "!=", True)
        d_i = np.array(g).dot(d_vector) / np.sum(g)

        # share p_i, p_j
        p_i = df_pivot.at[i[0], 'energy_share']
        p_j = df_pivot.at[i[1], 'energy_share']

        # contribution of combination i to stirling index
        s_i = d_i**alpha * (p_i*p_j)**beta

        s += s_i

    return s
