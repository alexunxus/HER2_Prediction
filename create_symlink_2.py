import csv
import os
import re

raw_label_file = "/mnt/cephrbd/data/A20009_CGMH_HER2/HER2_positive_2016-2019_compact.csv"
file_path_list = [
    "/mnt/cephrbd/data/A20009_CGMH_HER2/Image/20200310",
    "/mnt/cephrbd/data/A20009_CGMH_HER2/Image/20200317",
    "/mnt/cephrbd/data/A20009_CGMH_HER2/Image/20200323",
    "/mnt/cephrbd/data/A20009_CGMH_HER2/Image/20200415_Dr.Wu",
    "/mnt/cephrbd/data/A20009_CGMH_HER2/Image/20200420_Dr.Wu",
    "/mnt/cephrbd/data/A20009_CGMH_HER2/Image/20200422_Dr.Wu",
    "/mnt/cephrbd/data/A20009_CGMH_HER2/Image/20200608",
    "/mnt/cephrbd/data/A20009_CGMH_HER2/Image/20200618 Dr.wu",
    "/mnt/cephrbd/data/A20009_CGMH_HER2/Image/20200622",
]
output_path = "labeled_images"

os.makedirs(output_path, exist_ok=True)

with open(raw_label_file) as f:
    reader = csv.DictReader(f)
    content = []
    for row in reader:
        content.append(row)

for file_path in file_path_list:
    for file_name in os.listdir(file_path):
        short_file_name = re.findall("S[0-9]{4}-[0-9]{6}[A-Z]?", file_name)[0]
        print(file_name)
        full_path = os.path.join(file_path, file_name)

        file_idx = None
        for idx in range(len(content)):
            if content[idx]["patho_number"] == short_file_name:
                file_idx = idx

        if file_idx == None:
            print("{} is dangling.".format(full_path))
            continue

        her2_ihc = 3
        
        new_filename = short_file_name
        new_filename += "_IHC{}".format(her2_ihc)
        new_filename += ".ndpi"
        new_file_path = os.path.join(output_path, new_filename)

        if os.path.exists(new_file_path):
            print("File exists: {}".format(new_file_path))
            continue

        os.symlink(full_path, new_file_path)
