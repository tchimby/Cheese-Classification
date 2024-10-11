import pandas as pd

# To compare submissions (if one siubmission shows no big difference with the previous one, no need to submit it)
csv1 = pd.read_csv("/Data/hala.gamouh/cheese_classification_challenge/submission_ipadapt_tf2_4hrs_dalle.csv")
csv2 = pd.read_csv("/Data/hala.gamouh/cheese_classification_challenge/submission_ocr_ipadapt_dino.csv")

merged_df = pd.merge(csv1, csv2, on="id", suffixes=('_csv1', '_csv2'))

differences_count = (merged_df['label_csv1'] != merged_df['label_csv2']).sum()

print(f"Number of differences between the two files: {differences_count}")
