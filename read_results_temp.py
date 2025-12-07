import os

file_path = r'data/outputs/hawkes_results.txt'
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        print(f.read())
else:
    print("File not found")
