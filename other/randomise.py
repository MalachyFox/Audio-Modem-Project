import csv
import random

def add_noise_to_csv(file_name):
    with open(file_name, newline='') as infile:
        reader = csv.reader(infile)
        with open('output.csv', 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            for i in range(50):
                new_row = [random.uniform(-0.05, 0.05)]
                writer.writerow(new_row)
            for row in reader:
                writer.writerow(row)
            for i in range(50):
                new_row = [random.uniform(-0.05, 0.05)]
                writer.writerow(new_row)