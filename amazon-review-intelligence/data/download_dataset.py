import kagglehub

path = kagglehub.dataset_download(
    "dongrelaxman/amazon-reviews-dataset"
)

print("Path to dataset files:", path)