import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def count_files_by_gender(directory):
    pattern = re.compile(r'.*_(Female|Male|FEMALE|MALE)_(\d+)_\d+\.png$', re.IGNORECASE)

    gender_count = defaultdict(int)

    for file in os.listdir(directory):
        match = pattern.search(file)
        if match:
            gender = match.group(1).capitalize()
            gender_count[gender] += 1

    return gender_count

def plot_gender_distribution(gender_count):
    genders = list(gender_count.keys())
    counts = list(gender_count.values())

    plt.figure(figsize=(9, 6))
    ax = plt.bar(genders, counts, color=['skyblue', 'mediumseagreen'])
    plt.xlabel('성별', size=16)
    plt.ylabel('데이터 개수', size=16)
    plt.title('성별 데이터 개수 분포', size=16)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.bar_label(ax, size=16)
    plt.show()

directory = '../data/Test/'
gender_distribution = count_files_by_gender(directory)
for gender, count in sorted(gender_distribution.items()):
    print(f'{gender}: {count}장')

plot_gender_distribution(gender_distribution)
