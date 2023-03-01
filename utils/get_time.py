import itertools
import numpy as np
from pprint import pprint

domains = ["w", "d", "a"]
pairs = [[f"{d1}2{d2}" for d1 in domains] for d2 in domains]
pairs = list(itertools.chain.from_iterable(pairs))
pairs.sort()
# print(pairs)

ROUND_FACTOR = 2
all_accs_list = {}
files = [
    # "all.txt", 
    # "all2.txt", 
    # "all3.txt",
    # "all_step_scheduler.txt",
    # "all_source_trained_1.txt",
    "all_source_trained_2_specific_hp.txt",
]
for file in files:
    with open(file) as f:        
        name = None
        for line in f:
            splitted = line.split(" ")
            if splitted[0] == "##": # e.g. ## DANN
                name = splitted[1].strip()

            splitted = line.split(",") # a2w, acc1, acc2, 
            if splitted[0] in pairs:
                pair = splitted[0]
                name = name.lower()

                acc = float(splitted[5].strip()) #/ 60 / 60
                if name not in all_accs_list:
                    all_accs_list[name] = {p:[] for p in pairs}
                
                all_accs_list[name][pair].append(acc)

# all_accs_list format: {'dann': {'a2w': [acc1, acc2, acc3]}}

acc_means_by_model_name = {}
vars_by_model_name = {}
for name, pair_list in all_accs_list.items():
    accs = {p:[] for p in pairs}
    vars = {p:[] for p in pairs}
    for pair, acc_list in pair_list.items():
        if len(acc_list) > 0:
            ## Calculate average and round
            accs[pair] = round(sum(acc_list) / len(acc_list), ROUND_FACTOR)
            vars[pair] = round(np.var(acc_list) * 100, ROUND_FACTOR)
            print(vars[pair], "|||", acc_list)

        acc_means_by_model_name[name] = accs
        vars_by_model_name[name] = vars

# for name, acc_list in acc_means_by_model_name.items():
#     pprint(name)
# pprint(all_accs_list)
# pprint(acc_means_by_model_name)
# pprint(vars_by_model_name)

print()

latex_table = ""
header = [pair for pair in pairs if pair[0] != pair[-1]]
table = []
var_table = []

for name, acc_list in acc_means_by_model_name.items():
    if "target" in name or "source" in name:
        continue
    print("~~~~%%%%~~~", name)

    var_list = vars_by_model_name[name]
    valid_accs = []
    table_row = []
    var_table_row = []
    for pair in pairs:
        acc = acc_list[pair]
        var = var_list[pair]
        if pair[0] != pair[-1]: # exclude w2w, ...
            table_row.append(acc)
            var_table_row.append(var)
            if acc != None:
                valid_accs.append(acc)

    acc_average = round(sum(valid_accs) / len(header), ROUND_FACTOR)
    table_row.append(acc_average)
    table.append(table_row)

    var =round(np.var(valid_accs), ROUND_FACTOR)
    print(var, ">>>", valid_accs)
    var_table_row.append(var)
    var_table.append(var_table_row)

t = np.array(table)
t[t==None] = np.nan
# pprint(t)
col_max = t.min(axis=0)


pprint(table)
latex_table = ""
header = [pair for pair in pairs if pair[0] != pair[-1]]

name_map = {"base_source": "Source-Only"}
j = 0
for name, acc_list in acc_means_by_model_name.items():
    if "target" in name or "source" in name:
        continue

    latex_name = name
    if name in name_map:
        latex_name= name_map[name]
    latex_row = f"{latex_name.replace('_','-').upper()} &"
    acc_sum = 0
    for i, acc in enumerate(table[j]):  
        if i == len(table[j]) - 1:
            acc_str = f"${acc}$"
        else:
            acc_str = f"${acc}$"
        if acc == col_max[i]:
            latex_row += f" \\underline{{{acc_str}}} &"
        else:
            latex_row += f" {acc_str} &"
    latex_row = f"{latex_row[:-1]} \\\\ \hline"
    
    latex_table += f"{latex_row}\n"
    j += 1

print(*header, sep=" & ")
print(latex_table)

data = np.array(table)
legend = [key for key in acc_means_by_model_name.keys() if "source" not in key]
labels = [*header, "avg"]


# data = np.array([[71.75, 75.94 ,67.38, 90.99, 68.91, 96.67, 78.61], [64.0, 66.67, 37.32, 94.97, 45.74, 98.5, 67.87]])
# legend = ["CDAN", "Source-only"]
# labels = [*header, "avg"]

import matplotlib.pyplot as plt

# Assume your matrix is called 'data'
n, m = data.shape

# Create an array of x-coordinates for the bars
x = np.arange(m)

# Plot the bars for each row side by side
for i in range(n):

    row = data[i, :]
    plt.bar(x + (i-n/2)*0.1, row, width=0.08, align='center')

# Set x-axis tick labels and labels
plt.xticks(x, labels=labels)
# plt.xlabel("Task")
plt.ylabel("Time (s)")

# Add a legend

plt.legend(legend)

plt.show()
