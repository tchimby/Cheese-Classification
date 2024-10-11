import pandas as pd

# To merge the OCR and model predictions
submission_tf2 = pd.read_csv("/Data/hala.gamouh/cheese_classification_challenge/submission_ipadapt_valdata_15_epoch.csv")
ocr_results = pd.read_csv("/Data/hala.gamouh/cheese_classification_challenge/ocr_results.csv")

merged_df = pd.merge(submission_tf2, ocr_results, on="id", how="left")

def determine_label(row):
    if pd.notna(row['highest_similarity']) and row['highest_similarity'] > 0.65:
        return row['label_y']
    else:
        return row['label_x']

merged_df['label'] = merged_df.apply(determine_label, axis=1)

submission_ocr_custom_vision = merged_df[['id', 'label']]
submission_ocr_custom_vision.to_csv("/Data/hala.gamouh/cheese_classification_challenge/submission_ocr_ipadapt_dino6.csv", index=False)
