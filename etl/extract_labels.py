import re

def extract_labels(file_name):
    pattern = re.compile(r'.*_(Female|Male|FEMALE|MALE)_(\d+)_\d+\.png$', re.IGNORECASE)
    match = pattern.match(file_name)
    gender_str, age_str = match.groups()
    gender_label = 0 if gender_str.lower() == 'male' else 1
    age_label = int(age_str)

    return gender_label, age_label