import os


folder_path = '../data/Denoised_Test'
for filename in os.listdir(folder_path):
    if not filename.startswith('denoised'):
        new_filename = 'denoised_' + filename
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)
        os.rename(old_filepath, new_filepath)
        print(f'Renamed: {old_filepath} -> {new_filepath}')
