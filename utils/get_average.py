import itertools
import numpy as np
from pprint import pprint

domains = ["w", "d", "a"]
pairs = [[f"{d1}2{d2}" for d1 in domains] for d2 in domains]
pairs = list(itertools.chain.from_iterable(pairs))
pairs.sort()
# print(pairs)


all_accs_list = {}

for file in ["all_step_scheduler.txt", "all.txt"]:
    with open(file) as f:        
        name = None
        for line in f:
            splitted = line.split(" ")
            if splitted[0] == "##":
                name = splitted[1].strip()

            splitted = line.split(",")
            if splitted[0] in pairs:
                pair = splitted[0]
    
                acc = float(splitted[2].strip())
                if name not in all_accs_list:
                    all_accs_list[name] = {p:[] for p in pairs}
                
                all_accs_list[name][pair].append(acc)

acc_means = {}
for name, pair_list in all_accs_list.items():
    accs = {p:[] for p in pairs}
    for pair, acc_list in pair_list.items():
        if len(acc_list) > 0:
            ## Calculate average and round
            accs[pair] = round(100 * sum(acc_list) / len(acc_list), 2)

        acc_means[name] = accs

# for name, acc_list in acc_means.items():
#     pprint(name)
pprint(all_accs_list)
pprint(acc_means)

print()

latex_table = ""
header = [pair for pair in pairs if pair[0] != pair[-1]]
table = []

name_map = {}
for name, acc_list in acc_means.items():
    if "target" in name:
        continue

    acc_sum = 0
    table_row = []
    for pair in pairs:
        acc = acc_list[pair]
        if pair[0] != pair[-1]: # exclude w2w, ...
            table_row.append(acc)
            if acc != None:
                acc_sum += acc

    acc_average = round(acc_sum / len(header), 2)
    table_row.append(acc_average)
    table.append(table_row)

t = np.array(table)
t[t==None] = np.nan
# pprint(t)
col_max = t.max(axis=0)


latex_table = ""
header = [pair for pair in pairs if pair[0] != pair[-1]]

name_map = {}
j = 0
for name, acc_list in acc_means.items():
    if "target" in name:
        continue

    latex_name = name
    if name in name_map:
        latex_name= name_map[name]

    latex_row = f"{latex_name.replace('_','-').upper()} &"
    acc_sum = 0
    for i, acc in enumerate(table[j]):  
        if acc == col_max[i]:
            latex_row += f" \\underline{{${acc}$}} &"
        else:
            latex_row += f" ${acc}$ &"
    latex_row = f"{latex_row[:-1]} \\\\ \hline"
    
    latex_table += f"{latex_row}\n"
    j += 1

print(*header, sep=" & ")
print(latex_table)

