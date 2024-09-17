import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain
from collections import defaultdict
import pandas as pd
import numpy as np
from pycirclize import Circos
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from datetime import (
    datetime,
    # time,
    timedelta
)
from matplotlib.cm import get_cmap
import ast
import random
from scipy.spatial import distance
import scipy.cluster.hierarchy as sch
from trip_ids import (
    trip_ids_1,
    trip_ids_3,
    trip_ids_4,
    trip_ids_6,
    trip_ids_7,
    trip_ids_8,
    trip_ids_9,
    trip_ids_11,
    trip_ids_12,
    trip_ids_13,
    trip_ids_14,
    trip_ids_15,
    trip_ids_16,
    trip_ids_17,
    trip_ids_18,
    trip_ids_19,
)


def set_trip_ids(place_name):
    place_name = place_name.strip().lower()
    if place_name in trip_ids_1:
        return 1
    elif place_name in trip_ids_3:
        return 3
    elif place_name in trip_ids_4:
        return 4
    elif place_name in trip_ids_6:
        return 6
    elif place_name in trip_ids_7:
        return 7
    elif place_name in trip_ids_8:
        return 8
    elif place_name in trip_ids_9:
        return 9
    elif place_name in trip_ids_11:
        return 11
    elif place_name in trip_ids_12:
        return 12
    elif place_name in trip_ids_13:
        return 13
    elif place_name in trip_ids_14:
        return 14
    elif place_name in trip_ids_15:
        return 15
    elif place_name in trip_ids_16:
        return 16
    elif place_name in trip_ids_17:
        return 17
    elif place_name in trip_ids_18:
        return 18
    elif place_name in trip_ids_19:
        return 19
    else:
        return 97


def travel_time(combined_time):
    travel_time = None
    try:
        travel_time = 0
        # pst, pet = None, None
        pet = None
        for each in combined_time:
            s, e = each.split('-')
            e = e.strip('(next day))')
            st = datetime.strptime(s, '%I:%M %p')
            et = datetime.strptime(e, '%I:%M %p')
            if pet is None:
                # pst = st
                pet = et
                continue
            else:
                travel_time += (st - pet).seconds
                # pst = st
                pet = et
    except Exception as e:
        e
        travel_time = None
    return travel_time


def travel_times(combined_time):
    travel_times = []
    try:
        travel_time = 0
        # pst, pet = None, None
        pet = None
        for each in combined_time:
            s, e = each.split('-')
            e = e.strip('(next day))')
            st = datetime.strptime(s, '%I:%M %p')
            et = datetime.strptime(e, '%I:%M %p')
            if pet is None:
                # pst = st
                pet = et
                continue
            else:
                travel_time += (st - pet).seconds
                travel_times.append((st - pet).seconds)
                # pst = st
                pet = et
    except Exception as e:
        e
        travel_time = None
    return travel_times


def travel_time_original_data_chicago(combined_time):
    travel_time = None
    try:
        travel_time = 0
        # pst, pet = None, None
        pet = None
        for each in combined_time:
            s, e = each.split(' - ')
            e = e.strip()
            s = s.strip()
            st = datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
            et = datetime.strptime(e, '%Y-%m-%d %H:%M:%S')
            if pet is None:
                # pst = st
                pet = et
                continue
            else:
                travel_time += (st - pet).seconds
                # pst = st
                pet = et
    except Exception as e:
        e
        travel_time = None
    return travel_time


def travel_time_original_data(combined_time):
    travel_time = None
    try:
        travel_time = 0
        # pst, pet = None, None
        pet = None
        for each in combined_time:
            s, e = each.split('-')
            e = e.strip()
            s = int(s.strip())
            if e == 'end':
                e = '2359'
            e = int(float(e.strip()))
            if s == 0:
                s = '0000'
            if e == '0':
                e = '0000'
            if len(str(s)) not in [3, 4]:
                s = str(s)
                s = s.split('.')[0]
                s = int(s)
            if len(str(e)) not in [3, 4]:
                e = str(e)
                e = e.split('.')[0]
                e = int(e)
            if len(str(s)) == 3:
                a = f'{str(s)[0:1]}:{str(s)[1:]}'
                st = datetime.strptime(a, '%H:%M')
            elif len(str(s)) == 4:
                a = f'{str(s)[0:2]}:{str(s)[2:]}'
                st = datetime.strptime(f'{str(s)[0:2]}:{str(s)[2:]}', '%H:%M')
            else:
                raise Exception('Invalid time for s')
            if len(str(e)) == 3:
                a = f'{str(e)[0:1]}:{str(e)[1:]}'
                et = datetime.strptime(a, '%H:%M')
            elif len(str(e)) == 4:
                a = f'{str(e)[0:2]}:{str(e)[2:]}'
                et = datetime.strptime(f'{str(e)[0:2]}:{str(e)[2:]}', '%H:%M')
            else:
                raise Exception('Invalid time for e')
            if pet is None:
                # pst = st
                pet = et
                continue
            else:
                travel_time += (st - pet).seconds
                # pst = st
                pet = et
    except Exception as e:
        e
        travel_time = None
    return travel_time


def check_cyclic_path(G):
    '''
    Simple function to check if the start node and end node are same.
    '''
    start_node = None
    end_node = None
    for i, each in enumerate(G.edges()):
        if i == 0:
            start_node = each[0]
        end_node = each[1]
    if start_node == end_node:
        return True
    else:
        return False


def draw_graph(G):
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2000,
        node_color="skyblue",
        font_size=35,
        font_color="black"
    )
    labels = nx.get_edge_attributes(G, 'distance')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


def weighted_jaccard_similarity(dict1, dict2):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    intersection = keys1.intersection(keys2)
    union = keys1.union(keys2)

    numerator = sum(min(dict1[key], dict2[key]) for key in intersection)
    denominator = sum(
        max(dict1.get(key, 0), dict2.get(key, 0)) for key in union
    )

    if denominator == 0:
        return 0.0  # Avoid division by zero
    else:
        return numerator / denominator


def get_end_time(x):
    try:
        a = x['combined_time'][-1].split('-')[1]
        if 'AM' in a:
            t = a.split('AM')[0].strip()
            return f'{t} AM'
        elif 'PM' in a:
            t = a.split('PM')[0].strip()
            return f'{t} PM'
        else:
            return 'end'
    except Exception as e:
        e
        return 'end'


def get_digraph(x, key='loc_type'):
    G = nx.DiGraph()
    nodes = x[key]
    G.add_nodes_from(nodes)
    for i in range(len(nodes) - 1):
        G.add_edge(nodes[i], nodes[i + 1])
    for i in range(len(x[key])-1):
        G.add_edge(x[key][i], x[key][i+1])
    return G


def get_graphs_count(df):
    # Initialize a dictionary to store graph counts
    graph_counts = {}

    # Count the occurrences of each graph
    for graph in df['graph']:
        graph_str = nx.to_dict_of_dicts(graph)
        graph_str = str(graph_str)  # Convert graph to a stfor dic key
        if graph_str not in graph_counts.keys():
            graph_counts[graph_str] = 1
        else:
            graph_counts[graph_str] += 1
    return graph_counts


def get_combined_count_dictionary(dic1, dic2):
    # Initialize a combined dictionary with counts as tuples
    combined_dict = {}

    # Add counts from dict1
    for key, count in dic1.items():
        combined_dict[key] = (count, 0)

    # Add counts from dict2, or create a new entry if the key doesn't exist
    for key, count in dic2.items():
        if key in combined_dict:
            combined_dict[key] = (combined_dict[key][0], count)
        else:
            combined_dict[key] = (0, count)
    return combined_dict


def plot_counter_keys_count(
    counter1,
    counter2,
    counter1_label='Original Data',
    counter2_label='Generated Data',
    log=False,
):
    combined_keys = counter1.keys() | counter2.keys()
    combined_counter1 = {
        key: counter1[key] if key in counter1 else 0 for key in combined_keys
    }
    combined_counter2 = {
        key: counter2[key] if key in counter2 else 0 for key in combined_keys
    }

    # Extract keys and counts from combined counters
    keys = combined_keys
    counts1 = [combined_counter1[key] for key in keys]
    counts2 = [combined_counter2[key] for key in keys]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set bar width
    bar_width = 0.35

    # Set the positions of bars on the x-axis
    x = range(len(keys))

    # Create bars for each counter
    plt.bar(x, counts1, width=bar_width, label=counter1_label)
    plt.bar(
        [i + bar_width for i in x],
        counts2,
        width=bar_width,
        label=counter2_label
    )
    if log:
        plt.yscale('log')

    # Set labels and title
    plt.xlabel('Keys')
    plt.ylabel('Counts')
    plt.xticks([i + bar_width/2 for i in x], keys)
    plt.title('Number of times the graph is repeated')
    plt.xticks(rotation=90)
    # Add legend
    plt.legend()
    plt.legend(bbox_to_anchor=(0.69, 1))

    # Show the plot
    plt.show()


def get_chord_diagram(
        df,
        loc_type_key='loc_type',
        file_name='example.png',
        new_mappings=False,
        sim_mappings=False
):
    old_dnames = {
        1: "Home",
        2: "Work from Home",                   # Home
        3: "Work",                             # Work
        4: "Work-related",                     # Work
        5: "Volunteer",                        # Recreation
        6: "Drop-off/Pick-up",                 # Other
        7: "Change transport",                 # Other
        8: "School-student",                   # School
        9: "Child care",                       # Other
        10: "Adult care",                      # Other
        11: "Buy goods",                       # Grocery
        12: "Buy services",                    # Grocery
        13: "Buy meals",                       # Restaurant
        14: "General",                         # Other
        15: "Recreation",                      # Recreation
        16: "Exercise",                        # Recreation
        17: "Visit friends",                   # Other
        18: "Health care",                     # Other
        19: "Religious",                       # Other
        97: "Other",                           # Other
    }

    new_dnames = {
        1: "Home",
        2: "Work",
        3: "Community",
        4: "In Transit",
        5: "Education",
        6: "Care",
        7: "Shopping",
        8: "Eat Meal",
        9: "Recreational",
        10: "Social",
        11: "Other",
    }

    sim_dnames = {
        1: "Home",
        2: "Work",
        3: "Restaurant",
        4: "School",
        5: "Recreation",
        6: "Other",
    }

    if new_mappings is True:
        dnames = new_dnames
    elif sim_mappings is True:
        dnames = sim_dnames
    else:
        dnames = old_dnames
    C = []
    for lt in df[loc_type_key].values:
        for i, a in enumerate(lt):
            if i < len(lt)-1:
                C.append([a, lt[i+1]])

    counts = {}

    result_list = []

    for inner_list in C:
        inner_tuple = tuple(inner_list)

        if inner_tuple not in counts:
            counts[inner_tuple] = 1
        else:
            counts[inner_tuple] += 1
    for each in counts:
        result_list.append([each[0], each[1], counts[each]])

    list_of_codes = list(
        set(
            list(
                chain.from_iterable(df[loc_type_key].values)
            )
        )
    )
    if -7 in list_of_codes:
        list_of_codes.remove(-7)
    if -9 in list_of_codes:
        list_of_codes.remove(-9)
    C = pd.DataFrame(result_list, columns=['code_from', 'code_to', 'samples'])
    if -8 in list_of_codes:
        list_of_codes.remove(-8)
    dflow = defaultdict(lambda: {})
    for i in range(len(C)):
        dflow[C["code_from"][i]][C["code_to"][i]] = C["samples"][i]
    M = np.zeros((len(list_of_codes), len(list_of_codes)))
    for i in range(len(M)):
        for j in range(len(M)):
            try:
                M[i, j] = dflow[list_of_codes[i]][list_of_codes[j]]
            except KeyError:
                M[i, j] = 0
    M = pd.DataFrame(M, columns=[dnames[i] for i in list_of_codes])
    M.index = [dnames[i] for i in list_of_codes]

    circos = Circos.initialize_from_matrix(
        M,
        space=5,
        cmap="tab20",
        label_kws=dict(size=12, orientation="vertical"),
        link_kws=dict(ec="black", lw=0.05, direction=1),
    )
    circos.savefig(file_name)
    return C, M


def assign_proper_loc_type(row):
    final = None
    # print(row['uuid'])
    if type(row['loc_type']) is int:
        final = row['loc_type']
    elif row['loc_type'].isdigit():
        final = int(row['loc_type'])
    else:
        if ':' in row['loc_type']:
            if row['loc_type'][0] == '[':
                row['loc_type'] = row['loc_type'][1:-1]
            a = row['loc_type'].split(':')[0]
            final = int(a)
        else:
            a = row['loc_type'].split(' ')[0]
            if a.isdigit():
                final = int(a)
            else:
                a = row['loc_type'].lower()
                if 'home' in a:
                    final = 1
                elif 'recreational' in a:
                    final = 15
                elif 'exercise' in a:
                    final = 16
                elif 'meals' in a:
                    final = 13
                elif 'goods' in a:
                    if 'coffee' in a:
                        final = 13
                    else:
                        final = 11
                else:
                    final = 97
    if final in list(range(1, 20)):
        return final
    else:
        return 97


def assign_proper_loc_type_new_cat(row):
    final = None
    if type(row['loc_type']) is int:
        final = row['loc_type']
    elif row['loc_type'].isdigit():
        final = int(row['loc_type'])
    else:
        if ':' in row['loc_type']:
            if row['loc_type'][0] == '[':
                row['loc_type'] = row['loc_type'][1:-1]
            a = row['loc_type'].split(':')[0]
            final = int(a)
        else:
            a = row['loc_type'].split(' ')[0]
            if a.isdigit():
                final = int(a)
            else:
                final = 11
    if final in list(range(1, 11)):
        return final
    else:
        return 11


def get_acitivity_chain_plots(
    data,
    ax,
    show_legend=True,
    combined_time_key='combined_time'
):
    cmap = get_cmap('tab20')
    time_diffs = []
    labels = []
    event_plot_data = {
        '1': [],
        '2': [],
        '3': [],
        '4': [],
        '5': [],
        '6': [],
        '7': [],
        '8': [],
        '9': [],
        '10': [],
        '11': [],
        '12': [],
        '13': [],
        '14': [],
        '15': [],
        '16': [],
        '17': [],
        '18': [],
        '19': [],
        '97': [],
    }
    try:
        for i, (time_r, loc) in enumerate(
            zip(data[combined_time_key], data['loc_type2'])
        ):
            s, e = time_r.split('-')
            s = datetime.strptime(s, '%I:%M %p')
            e = datetime.strptime(e, '%I:%M %p')
            s = s.replace(year=2021, month=1, day=1)
            if len(data[combined_time_key]) == i+1:
                if 'AM' in time_r.split('-')[1]:
                    e = e.replace(
                        year=2021,
                        month=1,
                        day=1,
                        hour=23,
                        minute=59
                    )
                else:
                    e = e.replace(year=2021, month=1, day=1)
            else:
                e = e.replace(year=2021, month=1, day=1)
            t_diffs = mdates.drange(s, e, delta=timedelta(minutes=5))
            event_plot_data[str(loc)].extend(t_diffs)
            time_diffs.extend(t_diffs)
            for _ in range(len(t_diffs)):
                labels.append(str(loc))
    except Exception as e:
        for i, (time_r, loc) in enumerate(
            zip(data['time_combined'], data['loc_type'])
        ):
            s, e = time_r.split('-')
            s = s.strip()
            e = e.strip()
            if s == '0':
                s = '00:00'
            else:
                s = f'{s.split(".")[0][:-2]}:{s.split(".")[0][-2:]}'
                s = list(s)
                if len(s) == 4:
                    s.insert(0, '0')
                if s[0] == ' ':
                    s[0] = '0'
                s = ''.join(s)
            if e == 'end':
                e = '23:59'
            else:
                e = f'{e.split(".")[0][:-2]}:{e.split(".")[0][-2:]}'
                e = list(e)
                if e[0] == ' ':
                    e[0] = '0'
                e = ''.join(e)
            s = datetime.strptime(s, '%H:%M')
            e = datetime.strptime(e, '%H:%M')
            s = s.replace(year=2021, month=1, day=1)
            if len(data['time_combined']) == i+1:
                if 'AM' in time_r.split('-')[1]:
                    e = e.replace(
                        year=2021,
                        month=1,
                        day=1,
                        hour=23,
                        minute=59
                    )
                else:
                    e = e.replace(year=2021, month=1, day=1)
            else:
                e = e.replace(year=2021, month=1, day=1)
            t_diffs = mdates.drange(s, e, delta=timedelta(minutes=5))
            event_plot_data[str(loc)].extend(t_diffs)
            time_diffs.extend(t_diffs)
            for _ in range(len(t_diffs)):
                labels.append(str(loc))

    ax.eventplot(
        event_plot_data.values(),
        orientation='horizontal',
        label=event_plot_data.keys(),
        colors=cmap.colors
    )

    # Annotate the event markers
    for k, v in event_plot_data.items():
        if len(v) > 0:
            ax.annotate(
                '{}'.format(k),
                (v[0], int(k)),
                textcoords='offset points',
                xytext=(0, 1),
                ha='center'
            )

    # Format the x-axis to show time
    # ax.xticks(rotation=45)
    # ax.xaxis_date()
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

    ax.set_xlabel('Time')
    # ax.legend(event_plot_data.keys())
    if show_legend:
        ax.legend(
            event_plot_data.keys(),
            bbox_to_anchor=(1.05, 1.0),
            loc='upper left'
        )


def get_abstraction_graphs(grap_count, abs_location_types):
    new_counts = {}
    for k, v in grap_count.items():
        G = nx.DiGraph(ast.literal_eval(k))
        for each in abs_location_types:
            if each in G.nodes():
                in_nodes = [i[0] for i in G.in_edges(each)]
                out_nodes = [i[1] for i in G.out_edges(each)]
                G.remove_node(each)
                for i in in_nodes:
                    for j in out_nodes:
                        G.add_edge(i, j)
        key = str(nx.to_dict_of_dicts(G))
        if key not in new_counts.keys():
            new_counts[key] = v
        else:
            new_counts[key] += v
    return new_counts


def get_new_loc_type(row, key='loc_type2'):
    mappings = {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 3,
        6: 4,
        7: 4,
        8: 5,
        9: 6,
        10: 6,
        11: 7,
        12: 7,
        13: 8,
        14: 11,
        15: 9,
        16: 9,
        17: 10,
        18: 10,
        19: 3,
        97: 11,
    }
    if row[key] in mappings.keys():
        return mappings[row[key]]
    else:
        return 11


def get_new_loc_type_for_original_data(row, key='loc_type'):
    mappings = {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 3,
        6: 4,
        7: 4,
        8: 5,
        9: 6,
        10: 6,
        11: 7,
        12: 7,
        13: 8,
        14: 11,
        15: 9,
        16: 9,
        17: 10,
        18: 10,
        19: 3,
        97: 11,
    }
    final_vals = []
    for each in row[key]:
        if each in mappings.keys():
            final_vals.append(mappings[each])
        else:
            final_vals.append(11)
    return final_vals


def get_sim_loc_type_for_original_data(row, key='loc_type'):
    mappings = {
        1: 1,
        2: 1,               # Home
        3: 2,               # Work
        4: 2,               # Work
        5: 5,               # Recreation
        6: 6,               # Other
        7: 6,               # Other
        8: 4,               # School
        9: 6,               # Other
        10: 6,              # Other
        11: 6,              # Other
        12: 6,              # Other
        13: 3,              # Restaurant
        14: 6,              # Other
        15: 5,              # Recreation
        16: 5,              # Recreation
        17: 6,              # Other
        18: 6,              # Other
        19: 6,              # Other
        97: 6,              # Other
    }
    final_vals = []
    for each in row[key]:
        if each in mappings.keys():
            final_vals.append(mappings[each])
        else:
            final_vals.append(6)
    return final_vals


def get_new_loc_type_combined_df(row, key='loc_type2'):
    mappings = {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 3,
        6: 4,
        7: 4,
        8: 5,
        9: 6,
        10: 6,
        11: 7,
        12: 7,
        13: 8,
        14: 11,
        15: 9,
        16: 9,
        17: 10,
        18: 10,
        19: 3,
        97: 11,
    }
    res = []
    for each in row[key]:
        if each in mappings.keys():
            res.append(mappings[each])
        else:
            res.append(11)
    return res


def get_sim_loc_type_combined_df(row, key='loc_type2'):
    mappings = {
        1: 1,
        2: 1,               # Home
        3: 2,               # Work
        4: 2,               # Work
        5: 5,               # Recreation
        6: 6,               # Other
        7: 6,               # Other
        8: 4,               # School
        9: 6,               # Other
        10: 6,              # Other
        11: 6,              # Other
        12: 6,              # Other
        13: 3,              # Restaurant
        14: 6,              # Other
        15: 5,              # Recreation
        16: 5,              # Recreation
        17: 6,              # Other
        18: 6,              # Other
        19: 6,              # Other
        97: 6,              # Other
    }
    res = []
    for each in row[key]:
        if each in mappings.keys():
            res.append(mappings[each])
        else:
            res.append(11)
    return res


def get_normalized_norm(M1, M2):
    return np.linalg.norm(
        (M1/M1.sum().sum()).to_numpy() - (M2/M2.sum().sum()).to_numpy()
    )


def get_actual_survey_data(
        file_loc,
        new_loc_type=True,
        sim_loc_type=False
):
    data_t = pd.read_pickle(open(file_loc, 'rb'))
    if new_loc_type is True:
        data_t['loc_type'] = data_t.apply(
            lambda row: get_new_loc_type_for_original_data(row, 'loc_type'),
            axis=1
        )
    elif sim_loc_type is True:
        data_t['loc_type'] = data_t.apply(
            lambda row: get_sim_loc_type_for_original_data(row, 'loc_type'),
            axis=1
        )
    data_t['graph'] = data_t.apply(lambda x: get_digraph(x), axis=1)
    values_count = data_t['loc_type'].value_counts()
    relative_frequencies = values_count/data_t.shape[0]
    rank_relative_freq_list = list(enumerate(relative_frequencies.items(), 1))
    ranks, relative_freqs = zip(
        *[
            (rank, relative_freq)
            for rank, (category, relative_freq) in rank_relative_freq_list
        ]
    )
    return data_t, ranks, relative_freqs


def get_actual_survey_data_info(
    file_loc,
    name,
    new_loc_type=True,
    sim_loc_type=False
):
    print(name)
    print('*'*50)
    data_t, ranks, relative_freqs = get_actual_survey_data(
        f'/Users/prb977/Project/travel_survey_llm/{file_loc}',
        new_loc_type,
        sim_loc_type
    )
    print(f"Average Location: {data_t['location'].mean()}")
    print(f"Average Location: {data_t['location'].median()}")
    print(f'Number of samples: {data_t.shape[0]}')
    print(f'Travel time (Hrs): {data_t.travel_time.mean()/(60*60)}')
    print('*'*50)
    return data_t, ranks, relative_freqs


def get_generated_survey_data(
        file_loc,
        new_loc_type=True,
        sim_loc_type=False
):
    data_gen = pd.read_csv(file_loc)
    data_gen = data_gen[
        ~data_gen.uuid.isin(data_gen.loc[data_gen.loc_type.isna()].uuid)
    ]
    # 0. Assign proper location types (INT)
    data_gen['loc_type2'] = data_gen.apply(
        lambda row: assign_proper_loc_type(row),
        axis=1
    )
    # 1. Remove all the surveys with loc_type2 as 97
    data_gen = data_gen[
        ~data_gen.uuid.isin(data_gen.loc[data_gen.loc_type2 == 97].uuid)
    ]

    t = data_gen.copy()
    t['combined_time'] = t["arrival_time"] + '-' + t["departure_time"]
    t1 = t.groupby(['id'])['combined_time'].apply(list).reset_index()
    t2 = t.groupby(['id'])['loc_type2'].apply(list).reset_index()
    t = t1.merge(t2, on='id', how='inner')

    t['travel_times'] = t.apply(
        lambda x: travel_times(x['combined_time']),
        axis=1
    )
    t['travel_time'] = t.apply(
        lambda x: travel_time(x['combined_time']),
        axis=1
    )
    t_temp = t.copy()
    t_temp['combined_time_fixed'] = None
    t_temp['loc_type2_fixed'] = None
    for j, row in t_temp.iterrows():
        a = t_temp.iloc[j]

        loc_type2_fixed = []
        combined_time_fixed = []
        try:
            for i, each in enumerate(a.loc_type2):
                if i == 0:
                    loc_type2_fixed.append(each)
                    combined_time_fixed.append(a.combined_time[i])
                else:
                    if each == a.loc_type2[i-1]:
                        if a.combined_time[i] is np.nan or a.combined_time[i-1] is np.nan:  # noqa: E501
                            # 2. If the time is not available just remove
                            # those surveys
                            loc_type2_fixed = None
                            combined_time_fixed = None
                            break
                        f_start = a.combined_time[i-1].split('-')[0]
                        f_end = a.combined_time[i-1].split('-')[1]
                        f_end_t = datetime.strptime(f_end, '%I:%M %p')
                        s_start = a.combined_time[i].split('-')[0]
                        s_start_t = datetime.strptime(s_start, '%I:%M %p')
                        s_end = a.combined_time[i].split('-')[1]
                        diff = (s_start_t - f_end_t).total_seconds()

                        if diff > 60*60:
                            # 3. If the difference between two places is too
                            # high just remove those surveys
                            loc_type2_fixed.append(each)
                            combined_time_fixed.append(a.combined_time[i])
                            break
                        else:
                            combined_time_fixed[-1] = f'{f_start}-{s_end}'
                    else:
                        combined_time_fixed.append(a.combined_time[i])
                        loc_type2_fixed.append(each)
        except:  # noqa: E722
            loc_type2_fixed = None
            combined_time_fixed = None
        t_temp.at[j, 'combined_time_fixed'] = combined_time_fixed
        t_temp.at[j, 'loc_type2_fixed'] = loc_type2_fixed

    t = t_temp.loc[~t_temp.combined_time_fixed.isna()].reset_index(drop=True)
    t['loc_type2'] = t['loc_type2_fixed']
    t['combined_time'] = t['combined_time_fixed']
    t.drop(['loc_type2_fixed', 'combined_time_fixed'], axis=1, inplace=True)
    t['travel_times'] = t.apply(
        lambda x: travel_times(x['combined_time']),
        axis=1
    )
    t['travel_time'] = t.apply(
        lambda x: travel_time(x['combined_time']),
        axis=1
    )
    t['graph'] = t.apply(lambda x: get_digraph(x, 'loc_type2'), axis=1)

    if new_loc_type is True:
        t['loc_type_new'] = t.apply(
            lambda row: get_new_loc_type_combined_df(row, 'loc_type2'),
            axis=1
        )
    elif sim_loc_type is True:
        t['loc_type_new'] = t.apply(
            lambda row: get_sim_loc_type_combined_df(row, 'loc_type2'),
            axis=1
        )

    values_count = t['loc_type2'].value_counts()
    relative_frequencies = values_count/t.shape[0]
    rank_relative_freq_list = list(enumerate(relative_frequencies.items(), 1))
    ranks, relative_freqs = zip(
        *[
            (rank, relative_freq)
            for rank, (category, relative_freq) in rank_relative_freq_list
        ]
    )

    return t, ranks, relative_freqs


def get_generated_survey_data_no_preprocess(
        file_loc,
        new_loc_type=True,
        sim_loc_type=False
):
    data_gen = pd.read_csv(file_loc)
    data_gen = data_gen[
        ~data_gen.uuid.isin(data_gen.loc[data_gen.loc_type.isna()].uuid)
    ]
    data_gen['loc_type2'] = data_gen.apply(
        lambda row: assign_proper_loc_type(row),
        axis=1
    )

    t = data_gen.copy()
    t['combined_time'] = t["arrival_time"] + '-' + t["departure_time"]
    t1 = t.groupby(['id'])['combined_time'].apply(list).reset_index()
    t2 = t.groupby(['id'])['loc_type2'].apply(list).reset_index()
    t = t1.merge(t2, on='id', how='inner')

    t['travel_times'] = t.apply(
        lambda x: travel_times(x['combined_time']),
        axis=1
    )
    t['travel_time'] = t.apply(
        lambda x: travel_time(x['combined_time']),
        axis=1
    )
    t['graph'] = t.apply(lambda x: get_digraph(x, 'loc_type2'), axis=1)

    if new_loc_type is True:
        t['loc_type_new'] = t.apply(
            lambda row: get_new_loc_type_combined_df(row, 'loc_type2'),
            axis=1
        )
    elif sim_loc_type is True:
        t['loc_type_new'] = t.apply(
            lambda row: get_sim_loc_type_combined_df(row, 'loc_type2'),
            axis=1
        )

    values_count = t['loc_type2'].value_counts()
    relative_frequencies = values_count/t.shape[0]
    rank_relative_freq_list = list(enumerate(relative_frequencies.items(), 1))
    ranks, relative_freqs = zip(
        *[
            (rank, relative_freq)
            for rank, (category, relative_freq) in rank_relative_freq_list
        ]
    )

    return t, ranks, relative_freqs


def get_generated_survey_data_info(
    file_loc,
    name,
    new_loc_type=True,
    pre_process=True,
    sim_loc_type=False
):
    print(name)
    print('*'*50)
    if pre_process is True:
        t, ranks, relative_freqs = get_generated_survey_data(
            f'/Users/prb977/Project/travel_survey_llm/{file_loc}',
            new_loc_type,
            sim_loc_type
        )
    else:
        t, ranks, relative_freqs = get_generated_survey_data_no_preprocess(
            f'/Users/prb977/Project/travel_survey_llm/{file_loc}',
            new_loc_type,
            sim_loc_type
        )
    print(f"Average Location: {t.loc_type2.apply(lambda x: len(x)).mean()}")
    print(f"Median Location: {t.loc_type2.apply(lambda x: len(x)).median()}")
    print(f'Number of samples: {t.shape[0]}')
    print(f'Travel time: {t.travel_time.mean()/(60*60)}')
    print('*'*50)
    return t, ranks, relative_freqs


def fix_biases_from_trip_chains(t):
    t['loc_type_new'].iloc[0]
    for each in set([
        b
        for a in t.loc_type_new.values
        for b in a
    ]):
        t[f'count_{each}'] = 0
        t[f'original_count_{each}'] = 0
    for k, v in t.iterrows():
        for i, loc in enumerate(v['loc_type_new']):
            if i == 0:
                continue
            t.loc[k, f'count_{loc}'] += 1
            t.loc[k, f'original_count_{loc}'] += 1

    counts = t[['count_2', 'count_7', 'count_10']].sum(0)
    counts_required = counts.copy()
    counts_required['count_2'] = counts_required['count_2'] * 0.925
    counts_required['count_7'] = counts_required['count_7'] * 0.95
    counts_required['count_10'] = counts_required['count_10'] * 0.9775
    counts_required = counts_required.apply(lambda x: int(round(x, 0)))

    t_updated = t.copy()
    # 1. Get the differences in count for the required loc types
    count_diff = counts - counts_required
    count_diff.sort_values(ascending=True, inplace=True)
    count_diff

    # 2. Iterate through the difference in count starting from the smallest
    for key, val in count_diff.items():
        for i in range(val):
            rem_keys = [a for a in list(count_diff.keys()) if a != key]
            # 3. Get the rows where all loc type is present
            t_temp = t_updated[t_updated[f'{key}'] > 0]
            for k in rem_keys:
                t_temp = t_temp[t_temp[f'{k}'] > 0]
            if t_temp.shape[0] > 0:
                all = True
            else:
                # 4. If all loc type is not present, get the rows where the
                # loc type is present
                t_temp = t_updated[t_updated[f'{key}'] > 0]
                all = False
            # 5. Remove the loc type once from a random row
            change_id = t_temp.iloc[random.randint(0, t_temp.shape[0]-1)].id
            t_updated.loc[t_updated['id'] == change_id, key] -= 1
            if all:
                for k in rem_keys:
                    t_updated.loc[t_updated['id'] == change_id, k] -= 1

            count_diff[key] -= 1
            count_diff = count_diff[count_diff > 0]
    for each in set([
        b
        for a in t_updated.loc_type_new.values
        for b in a
    ]):
        t_updated[f'count_diff_{each}'] = t_updated[f'original_count_{each}']\
              - t_updated[f'count_{each}']

    for each in set([
        b
        for a in t_updated.loc_type_new.values
        for b in a
    ]):
        t_updated = t_updated.loc[t_updated[f'count_diff_{each}'] == 0]

    t = t_updated.copy()

    values_count = t['loc_type_new'].value_counts()
    relative_frequencies = values_count/t.shape[0]
    rank_relative_freq_list = list(enumerate(relative_frequencies.items(), 1))
    ranks, relative_freqs = zip(
        *[
            (rank, relative_freq)
            for rank, (category, relative_freq) in rank_relative_freq_list
        ]
    )

    print(f"Average Location: {t.loc_type_new.apply(lambda x: len(x)).mean()}")
    print(
        f"Median Location: {t.loc_type_new.apply(lambda x: len(x)).median()}"
    )
    print(f'Number of samples: {t.shape[0]}')
    print(f'Travel time: {t.travel_time.mean()/(60*60)}')
    print('*'*50)
    return t, ranks, relative_freqs


def get_transient_prob_1(t, loc_type_key):
    dic = {}
    for each in t[loc_type_key]:
        for i in range(len(each)-1):
            k = each[i:i+2]
            if str(k) in dic.keys():
                dic[str(k)] += 1
            else:
                dic[str(k)] = 1
    main_lis = []
    for i in range(1, 12):
        sub_lis = []
        for j in range(1, 12):
            if str([i, j]) in dic.keys():
                sub_lis.append(dic[str([i, j])])
            else:
                sub_lis.append(0)
        main_lis.append(sub_lis)
    main_lis

    cols = [
        "Home",
        "Work",
        "Community",
        "In Transit",
        "Education",
        "Care",
        "Shopping",
        "Eat Meal",
        "Recreational",
        "Social",
        "Other",
    ]

    df = pd.DataFrame(main_lis, columns=cols, index=cols)
    df
    return df


def get_transient_prob_1_sim(t, loc_type_key):
    dic = {}
    for each in t[loc_type_key]:
        for i in range(len(each)-1):
            k = each[i:i+2]
            if str(k) in dic.keys():
                dic[str(k)] += 1
            else:
                dic[str(k)] = 1
    main_lis = []
    for i in range(1, 7):
        sub_lis = []
        for j in range(1, 7):
            if str([i, j]) in dic.keys():
                sub_lis.append(dic[str([i, j])])
            else:
                sub_lis.append(0)
        main_lis.append(sub_lis)
    main_lis

    cols = [
        "Home",
        "Work",
        "Restaurant",
        "School",
        "Recreation",
        "Other"
    ]

    df = pd.DataFrame(main_lis, columns=cols, index=cols)
    df
    return df


def get_transient_prob_2(t, loc_type_key):
    dic = {}
    for each in t[loc_type_key]:
        for i in range(len(each)-2):
            k = each[i:i+3]
            if str(k) in dic.keys():
                dic[str(k)] += 1
            else:
                dic[str(k)] = 1
    main_lis = []
    for i in range(1, 12):
        for j in range(1, 12):
            sub_lis = []
            for k in range(1, 12):
                if str([i, j, k]) in dic.keys():
                    sub_lis.append(dic[str([i, j, k])])
                else:
                    sub_lis.append(0)

            sub_lis.insert(0, j)
            sub_lis.insert(0, i)
            main_lis.append(sub_lis)
    main_lis

    df = pd.DataFrame(
        main_lis,
        columns=[
            'xt-2', 'xt-1', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        ]
    )
    df.set_index(['xt-2', 'xt-1'], inplace=True)
    return df


def get_transient_prob_2_sim(t, loc_type_key):
    dic = {}
    for each in t[loc_type_key]:
        for i in range(len(each)-2):
            k = each[i:i+3]
            if str(k) in dic.keys():
                dic[str(k)] += 1
            else:
                dic[str(k)] = 1
    main_lis = []
    for i in range(1, 7):
        for j in range(1, 7):
            sub_lis = []
            for k in range(1, 7):
                if str([i, j, k]) in dic.keys():
                    sub_lis.append(dic[str([i, j, k])])
                else:
                    sub_lis.append(0)

            sub_lis.insert(0, j)
            sub_lis.insert(0, i)
            main_lis.append(sub_lis)
    main_lis

    df = pd.DataFrame(
        main_lis,
        columns=[
            'xt-2', 'xt-1', 1, 2, 3, 4, 5, 6
        ]
    )
    df.set_index(['xt-2', 'xt-1'], inplace=True)
    return df


def get_transient_prob_3(t, loc_type_key):
    dic = {}
    for each in t[loc_type_key]:
        for i in range(len(each)-3):
            k = each[i:i+4]
            if str(k) in dic.keys():
                dic[str(k)] += 1
            else:
                dic[str(k)] = 1
    main_lis = []
    for i in range(1, 12):
        for j in range(1, 12):
            for k in range(1, 12):
                sub_lis = []
                for p in range(1, 12):
                    if str([i, j, k, p]) in dic.keys():
                        sub_lis.append(dic[str([i, j, k, p])])
                    else:
                        sub_lis.append(0)

                sub_lis.insert(0, k)
                sub_lis.insert(0, j)
                sub_lis.insert(0, i)
                main_lis.append(sub_lis)

    df = pd.DataFrame(
        main_lis,
        columns=[
            'xt-3', 'xt-2', 'xt-1', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        ]
    )
    df.set_index(['xt-3', 'xt-2', 'xt-1'], inplace=True)
    return df


def get_transient_prob_4(t, loc_type_key):
    dic = {}
    for each in t[loc_type_key]:
        for i in range(len(each)-4):
            k = each[i:i+5]
            if str(k) in dic.keys():
                dic[str(k)] += 1
            else:
                dic[str(k)] = 1
    main_lis = []
    for i in range(1, 12):
        for j in range(1, 12):
            for k in range(1, 12):
                for p in range(1, 12):
                    sub_lis = []
                    for m in range(1, 12):
                        if str([i, j, k, p, m]) in dic.keys():
                            sub_lis.append(dic[str([i, j, k, p, m])])
                        else:
                            sub_lis.append(0)

                    sub_lis.insert(0, p)
                    sub_lis.insert(0, k)
                    sub_lis.insert(0, j)
                    sub_lis.insert(0, i)
                    main_lis.append(sub_lis)

    df = pd.DataFrame(
        main_lis,
        columns=[
            'xt-4', 'xt-3', 'xt-2', 'xt-1', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        ]
    )
    df.set_index(['xt-4', 'xt-3', 'xt-2', 'xt-1'], inplace=True)
    return df


def get_transient_prob_5(t, loc_type_key):
    dic = {}
    for each in t[loc_type_key]:
        for i in range(len(each)-5):
            k = each[i:i+6]
            if str(k) in dic.keys():
                dic[str(k)] += 1
            else:
                dic[str(k)] = 1
    main_lis = []
    for i in range(1, 12):
        for j in range(1, 12):
            for k in range(1, 12):
                for p in range(1, 12):
                    for m in range(1, 12):
                        sub_lis = []
                        for n in range(1, 12):
                            if str([i, j, k, p, m, n]) in dic.keys():
                                sub_lis.append(dic[str([i, j, k, p, m, n])])
                            else:
                                sub_lis.append(0)

                        sub_lis.insert(0, m)
                        sub_lis.insert(0, p)
                        sub_lis.insert(0, k)
                        sub_lis.insert(0, j)
                        sub_lis.insert(0, i)
                        main_lis.append(sub_lis)

    df = pd.DataFrame(
        main_lis,
        columns=[
            'xt-5',
            'xt-4',
            'xt-3',
            'xt-2',
            'xt-1',
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        ]
    )
    df.set_index(['xt-5', 'xt-4', 'xt-3', 'xt-2', 'xt-1'], inplace=True)
    return df


def get_datadf_from_transient_probs(order=1, model='gemini'):
    data = {
        'SF_Original': [
            0,
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
        ],
        'SF_Generated': [
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_sf'),
                    eval(f't_prob_{order}_sf_{model}')
                ),
                5
            ),
            0,
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
        ],
        'DC_Original': [
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_sf'),
                    eval(f't_orig_prob_{order}_dc'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_sf_{model}'),
                    eval(f't_orig_prob_{order}_dc'),
                ),
                5
            ),
            0,
            '',
            '',
            '',
            '',
            '',
            '',
            '',
        ],
        'DC_Generated': [
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_sf'),
                    eval(f't_prob_{order}_dc_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_sf_{model}'),
                    eval(f't_prob_{order}_dc_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_dc'),
                    eval(f't_prob_{order}_dc_{model}'),
                ),
                5
            ),
            0,
            '',
            '',
            '',
            '',
            '',
            '',
        ],
        'DFW_Original': [
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_sf'),
                    eval(f't_orig_prob_{order}_dfw'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_sf_{model}'),
                    eval(f't_orig_prob_{order}_dfw'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_dc'),
                    eval(f't_orig_prob_{order}_dfw'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_dc_{model}'),
                    eval(f't_orig_prob_{order}_dfw'),
                ),
                5
            ),
            0,
            '',
            '',
            '',
            '',
            '',
        ],
        'DFW_Generated': [
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_sf'),
                    eval(f't_prob_{order}_dfw_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_sf_{model}'),
                    eval(f't_prob_{order}_dfw_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_dc'),
                    eval(f't_prob_{order}_dfw_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_dc_{model}'),
                    eval(f't_prob_{order}_dfw_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_dfw_{model}'),
                    eval(f't_orig_prob_{order}_dfw'),
                ),
                5
            ),
            0,
            '',
            '',
            '',
            '',
        ],
        'LA_Original': [
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_sf'),
                    eval(f't_orig_prob_{order}_la'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_sf_{model}'),
                    eval(f't_orig_prob_{order}_la'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_dc'),
                    eval(f't_orig_prob_{order}_la'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_dc_{model}'),
                    eval(f't_orig_prob_{order}_la'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_dfw'),
                    eval(f't_orig_prob_{order}_la'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_dfw_{model}'),
                    eval(f't_orig_prob_{order}_la'),
                ),
                5
            ),
            0,
            '',
            '',
            '',
        ],
        'LA_Generated': [
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_sf'),
                    eval(f't_prob_{order}_la_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_sf_{model}'),
                    eval(f't_prob_{order}_la_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_dc'),
                    eval(f't_prob_{order}_la_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_dc_{model}'),
                    eval(f't_prob_{order}_la_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_dfw'),
                    eval(f't_prob_{order}_la_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_dfw_{model}'),
                    eval(f't_prob_{order}_la_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_la'),
                    eval(f't_prob_{order}_la_{model}'),
                ),
                5
            ),
            '',
            '',
            '',
        ],
        'Minneapolis_Original': [
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_sf'),
                    eval(f't_orig_prob_{order}_minneapolis'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_sf_{model}'),
                    eval(f't_orig_prob_{order}_minneapolis'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_dc'),
                    eval(f't_orig_prob_{order}_minneapolis'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_dc_{model}'),
                    eval(f't_orig_prob_{order}_minneapolis'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_dfw'),
                    eval(f't_orig_prob_{order}_minneapolis'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_dfw_{model}'),
                    eval(f't_orig_prob_{order}_minneapolis'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_la'),
                    eval(f't_orig_prob_{order}_minneapolis'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_la_{model}'),
                    eval(f't_orig_prob_{order}_minneapolis'),
                ),
                5
            ),
            0,
            '',
        ],
        'Minneapolis_Generated': [
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_sf'),
                    eval(f't_prob_{order}_minneapolis_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_sf_{model}'),
                    eval(f't_prob_{order}_minneapolis_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_dc'),
                    eval(f't_prob_{order}_minneapolis_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_dc_{model}'),
                    eval(f't_prob_{order}_minneapolis_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_dfw'),
                    eval(f't_prob_{order}_minneapolis_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_dfw_{model}'),
                    eval(f't_prob_{order}_minneapolis_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_la'),
                    eval(f't_prob_{order}_minneapolis_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_prob_{order}_la_{model}'),
                    eval(f't_prob_{order}_minneapolis_{model}'),
                ),
                5
            ),
            round(
                get_normalized_norm(
                    eval(f't_orig_prob_{order}_minneapolis'),
                    eval(f't_prob_{order}_minneapolis_{model}'),
                ),
                5
            ),
            '',
        ]
    }
    cell_colours = [
        ['white', 'lightcoral', 'lightgreen', 'cyan', 'lightgreen', 'cyan', 'lightgreen', 'cyan', 'lightgreen', 'cyan'],  # noqa: E501
        ['white', 'white', 'cyan', 'orange', 'cyan', 'orange', 'cyan', 'orange', 'cyan', 'orange'],  # noqa: E501
        ['white', 'white', 'white', 'lightcoral', 'lightgreen', 'cyan', 'lightgreen', 'cyan', 'lightgreen', 'cyan'],  # noqa: E501
        ['white', 'white', 'white', 'white', 'cyan', 'orange', 'cyan', 'orange', 'cyan', 'orange'],  # noqa: E501
        ['white', 'white', 'white', 'white', 'white', 'lightcoral', 'lightgreen', 'cyan', 'lightgreen', 'cyan'],  # noqa: E501
        ['white', 'white', 'white', 'white', 'white', 'white', 'cyan', 'orange', 'cyan', 'orange'],  # noqa: E501
        ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'lightcoral', 'lightgreen', 'cyan'],  # noqa: E501
        ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'cyan', 'orange'],  # noqa: E501
        ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'lightcoral'],  # noqa: E501
        ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white'],  # noqa: E501
    ]

    datadf = pd.DataFrame(data, index=data.keys())

    SOG = [datadf.values[i][j] for i, j in list(zip(*np.where(np.array(cell_colours) == 'lightcoral')))]  # noqa: E501
    DOO = [datadf.values[i][j] for i, j in list(zip(*np.where(np.array(cell_colours) == 'lightgreen')))]  # noqa: E501
    DOG = [datadf.values[i][j] for i, j in list(zip(*np.where(np.array(cell_colours) == 'cyan')))]  # noqa: E501
    DGG = [datadf.values[i][j] for i, j in list(zip(*np.where(np.array(cell_colours) == 'orange')))]  # noqa: E501

    datadf
    return SOG, DOO, DOG, DGG, datadf


def plot_table_inter_city_norm_2(
        df,
        SOG,
        DOO,
        DOG,
        DGG
):
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    cell_colours = [
        ['white', 'lightcoral', 'lightgreen', 'cyan', 'lightgreen', 'cyan', 'lightgreen', 'cyan', 'lightgreen', 'cyan'],  # noqa: E501
        ['white', 'white', 'cyan', 'orange', 'cyan', 'orange', 'cyan', 'orange', 'cyan', 'orange'],  # noqa: E501
        ['white', 'white', 'white', 'lightcoral', 'lightgreen', 'cyan', 'lightgreen', 'cyan', 'lightgreen', 'cyan'],  # noqa: E501
        ['white', 'white', 'white', 'white', 'cyan', 'orange', 'cyan', 'orange', 'cyan', 'orange'],  # noqa: E501
        ['white', 'white', 'white', 'white', 'white', 'lightcoral', 'lightgreen', 'cyan', 'lightgreen', 'cyan'],  # noqa: E501
        ['white', 'white', 'white', 'white', 'white', 'white', 'cyan', 'orange', 'cyan', 'orange'],  # noqa: E501
        ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'lightcoral', 'lightgreen', 'cyan'],  # noqa: E501
        ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'cyan', 'orange'],  # noqa: E501
        ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'lightcoral'],  # noqa: E501
        ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white'],  # noqa: E501
    ]
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc='center',
        cellColours=cell_colours)
    # table.set_fontsize(5)
    table.scale(3, 3)
    fig.tight_layout()
    red_patch = mpatches.Patch(
        color='lightcoral',
        label='Original & Generated of Same City'
    )
    green_patch = mpatches.Patch(
        color='lightgreen',
        label='Original & Original of Different City'
    )
    cyan_patch = mpatches.Patch(
        color='cyan',
        label='Original & Generated of Different City'
    )
    orange_patch = mpatches.Patch(
        color='orange',
        label='Generated & Generated of Different City'
    )
    plt.legend(
        handles=[red_patch, green_patch, cyan_patch, orange_patch],
        loc='lower center'
    )
    plt.show()

    plt.boxplot(
        [SOG, DOO, DOG, DGG],
        labels=[
            'Original & Generated of Same City',
            'Original & Original of Different City',
            'Original & Generated of Different City',
            'Generated & Generated of Different City',
        ]
    )
    plt.xticks(rotation=45)
    plt.show()


def create_plots_for_destination_prob(res):
    res.transpose().plot.bar(
        stacked=True,
        title='Destination Probabilities',
        cmap='tab20'
    )
    plt.tight_layout()
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

    res.plot.bar(
        title='Destination Probabilities',
        figsize=(14, 5),
        cmap='tab20'
    )
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

    pd.concat(
        [
            res[['SF']].sub(res['SF_orig'], axis=0),
            res[['DC']].sub(res['DC_orig'], axis=0),
            res[['DFW']].sub(res['DFW_orig'], axis=0),
            res[['LA']].sub(res['LA_orig'], axis=0),
            res[['Minneapolis']].sub(res['Minneapolis_orig'], axis=0),
        ],
        axis=1
    ).plot.bar(
        title='Difference in Destination Probabilities (Generated - Origial)',
        figsize=(14, 5),
        cmap='tab20'
    )
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

    pd.concat(
        [
            res[['SF']].sub(res['SF_orig'], axis=0)*100,
            res[['DC']].sub(res['DC_orig'], axis=0)*100,
            res[['DFW']].sub(res['DFW_orig'], axis=0)*100,
            res[['LA']].sub(res['LA_orig'], axis=0)*100,
            res[['Minneapolis']].sub(res['Minneapolis_orig'], axis=0)*100,
        ],
        axis=1
    ).mean(axis=1).plot.bar(
        title='Average Difference in Destination Probabilities (Generated - Origial)',  # noqa: E501
        figsize=(14, 5),
        cmap='tab20',
    )
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

    vals = pd.concat(
        [
            res[['SF']].sub(res['SF_orig'], axis=0)*100,
            res[['DC']].sub(res['DC_orig'], axis=0)*100,
            res[['DFW']].sub(res['DFW_orig'], axis=0)*100,
            res[['LA']].sub(res['LA_orig'], axis=0)*100,
            res[['Minneapolis']].sub(res['Minneapolis_orig'], axis=0)*100,
        ],
        axis=1
    ).mean(axis=1)
    print(vals)
    # vals_abs = np.abs(list(vals))
    print(np.mean(np.abs(vals)))
    print(np.percentile(np.abs(vals), [25, 50, 75, 90]))
    print(np.percentile(np.abs(vals), 75))
    print(
        vals.where(
            np.abs(vals) > (np.percentile(np.abs(vals), 75))
        ).dropna()
    )
    return vals


def plotting_the_dendogram(data, label_list, method="ward"):

    # Calculate the linkage matrix
    Z = sch.linkage(data, method=method, optimal_ordering=True)

    # Override the default linewidth.
    plt.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                           figsize=(4, 3))

    # --- Plotting the dendogram
    S = sch.dendrogram(
        Z,
        ax=ax,
        labels=label_list,
        orientation="right"
    )
    return S


def plotting_the_dendogram_JS(matrices):
    # Compute pairwise Jensen-Shannon distances
    distances = []
    for i in range(len(matrices)):
        for j in range(i+1, len(matrices)):
            mat1 = matrices[i]
            mat2 = matrices[j]
            js_dist = np.mean(
                [distance.jensenshannon(mat1[i], mat2[i]) for i in range(11)]
            )
            distances.append(js_dist)

    # Perform hierarchical clustering
    linked = sch.linkage(distances, method='complete')

    # Plot the dendrogram
    plt.figure(figsize=(4, 3))
    sch.dendrogram(
        linked,
        truncate_mode='level',
        p=3,
        show_leaf_counts=True,
        labels=['SF', 'DC', 'DFW', 'LA', 'Minneapolis'],
        orientation='right'
    )
    # plt.xlabel('Transition Matrix')
    plt.xlabel('Jensen-Shannon Distance')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.show()

    # Obtain cluster assignments
    cluster_assignments = sch.fcluster(linked, t=0.5, criterion='distance')
    print('Cluster assignments:', cluster_assignments)
