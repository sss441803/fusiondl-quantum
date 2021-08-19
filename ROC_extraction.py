file1 = open('torch_learn.out', 'r')
for line in file1:
    if 'Validation ROC' in line:
        print(line[:-1])