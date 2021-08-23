file1 = open('c4c4c3c2classical.out', 'r')
for line in file1:
    if 'Validation ROC' in line:
        print(line[:-1])