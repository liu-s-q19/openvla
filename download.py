import tensorflow_datasets as tfds

# import tfds_nightly as tfds
import tqdm

# optionally replace the DATASET_NAMES below with the list of filtered datasets from the google sheet
DATASET_NAMES = [
    "berkeley_gnm_cory_hall",
]
DOWNLOAD_DIR = "/root/autodl-tmp/openvla/datasets"

print(f"Downloading {len(DATASET_NAMES)} datasets to {DOWNLOAD_DIR}.")
for dataset_name in tqdm.tqdm(DATASET_NAMES):
    _ = tfds.load(dataset_name, data_dir=DOWNLOAD_DIR)
