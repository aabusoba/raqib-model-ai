import kagglehub
import os

path = kagglehub.dataset_download("tsiaras/uk-road-safety-accidents-and-vehicles")
print(f"Dataset downloaded to: {path}")
print("Files in the dataset:")
for root, dirs, files in os.walk(path):
    for file in files:
        print(os.path.join(root, file))
