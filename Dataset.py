import numpy as np
import pandas as pd
import os
import csv
import numpy as np

def combine(data):
    data_lines = data.split('\n')
    tourist_places = {}
    current_tourist_place = None
    current_captions = []

    for line in data_lines:
        if line.startswith("str(i)+/n"):  # Replace str(i)+/n with actual values of i
            if current_tourist_place is not None:
                tourist_places[current_tourist_place] = current_captions
            current_tourist_place = line[9:]
            current_captions = []
        elif line.startswith("https://"):
            caption = line.split('<>')[1]
            current_captions.append(caption)

    # Add the last tourist place and captions
    if current_tourist_place is not None:
        tourist_places[current_tourist_place] = current_captions

    with open('new_dataset.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        for place, captions in tourist_places.items():
            # Pad the captions list with NaN values to make it 61 columns long
            padded_captions = captions + [np.nan] * (61 - len(captions))
            writer.writerow([place] + padded_captions)

folder_path = "full"  # Replace this with the path to your folder
file_contents = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):  # Assuming you want to read only .txt files
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as file:
            content = file.read()
            file_contents.append(content)
for i in file_contents:
    combine(i)
    
