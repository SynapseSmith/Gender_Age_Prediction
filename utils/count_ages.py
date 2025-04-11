import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def count_files_by_age(directory):
    pattern = re.compile(r'.*_(Female|Male|FEMALE|MALE)_(\d+)_\d+\.png$', re.IGNORECASE)

    age_count = defaultdict(int)

    for file in os.listdir(directory):
        match = pattern.match(file)
        if match:
            age = int(match.group(2))
            age_group = (age // 10) * 10
            age_count[age_group] += 1

    return age_count

def plot_age_distribution(age_count):
    ages = list(age_count.keys())
    counts = list(age_count.values())

    colormap = plt.get_cmap('tab20')
    colors = [colormap(i / len(ages)) for i in range(len(ages))]

    plt.figure(figsize=(9, 6))
    ax = plt.bar(ages, counts, width=6, color=colors)
    plt.xlabel('나이대', size=16)
    plt.ylabel('데이터 개수', size=16)
    plt.title('나이대별 데이터 개수 분포', size=16)
    plt.xticks([x for x in range(0, 100, 10)], [f'{x}대' for x in range(0, 100, 10)], size=16)
    plt.yticks(size=16)
    plt.xlim([-10, 100])
    plt.bar_label(ax, size=16)
    plt.show()

directory = '../data/Copied_Test2/'
age_distribution = count_files_by_age(directory)
for age_group, count in sorted(age_distribution.items()):
    print(f'{age_group}대: {count}장')

plot_age_distribution(age_distribution)