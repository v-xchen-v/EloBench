import csv

# Read the text file and split into lines
with open(r'data/quora_100/100_question_quora.txt', 'r') as file:
    lines = file.read().splitlines()

# Write to a CSV file
with open(r'100_question_quora.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for item in lines:
        writer.writerow([item])
