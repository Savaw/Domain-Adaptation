import itertools

domains = ["w", "d", "a"]
pairs = [[f"{d1}2{d2}" for d1 in domains] for d2 in domains]
pairs = list(itertools.chain.from_iterable(pairs))
pairs.sort()
print(pairs)

acc_means = {}

with open("all.txt") as f:
    accs_list = {p:[] for p in pairs}
    name = None
    for line in f:
        splitted = line.split(" ")
        if splitted[0] == "##":
            if name:
                accs = {p:None for p in pairs}
                for pair, acc_list in accs_list.items():
                    if len(acc_list) > 0:
                        ## Calculate average and round
                        accs[pair] = round(100 * sum(acc_list) / len(acc_list), 2)

                acc_means[name] = accs
                accs_list = {p:[] for p in pairs}

            name = splitted[1].strip()
        splitted = line.split(",")
        if splitted[0] in pairs:
            pair = splitted[0]
            acc = float(splitted[2].strip())
            accs_list[pair].append(acc)


for name, acc_list in acc_means.items():
    print(name)
    print(acc_list)

print()
latex_table = ""
header = [pair for pair in pairs if pair[0] != pair[-1]]

for name, acc_list in acc_means.items():
    if "target" in name:
        continue
    latex_row = f"{name.replace('_','-').upper()} &"
    acc_sum = 0
    for pair in pairs:
        acc = acc_list[pair]
        if pair[0] != pair[-1]: # exclude w2w, ...
            latex_row += f" ${acc}$ &"
            if acc != None:
                acc_sum += acc
    acc_average = round(acc_sum / len(header), 2)
    latex_row = f"{latex_row} ${acc_average}$ \\\\ \hline"
    
    latex_table += f"{latex_row}\n"

print(*header, sep=" & ")
print(latex_table)

