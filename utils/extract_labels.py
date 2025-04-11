import re


def extract_labels(file_name):
    match = re.search(r'(Female|Male|FEMALE|MALE)_(\d+)', file_name)

    gender = match.group(1)
    age = int(match.group(2))

    gender = 0 if gender.lower() == 'male' else 1

    return gender, age