file1 = open('outputs/c4c3k3.out', 'r')
for line in file1:
    if 'Validation ROC' in line:
        print(line[16:-1])