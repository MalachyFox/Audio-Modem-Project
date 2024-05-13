import csv
import random

file_name = 'weekend-challenge/file1.csv'

with open(file_name, newline='') as infile:
    reader = csv.reader(infile)
    with open('output.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        for i in range(5):
            new_row = [random.uniform(-1, 1)]
            writer.writerow(new_row)
        for row in reader:
            modified_row = [float(value) + random.uniform(-1, 1) for value in row]
            writer.writerow(modified_row)
        for i in range(5):
            new_row = [random.uniform(-1, 1)]
            writer.writerow(new_row)