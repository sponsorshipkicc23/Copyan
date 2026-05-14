import pandas as pd
import os
from sklearn.feature_selection import mutual_info_classif

def select_features_mi(df, result_dir, patient_name, target_col="IDA_Label", threshold=0.01):
    exclude_cols = ["Cell_Label", "X", "Y", target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X_feat = df[feature_cols].fillna(0)
    y_feat = df[target_col]
    
    mi_scores = mutual_info_classif(X_feat, y_feat, random_state=42)
    mi_results = pd.DataFrame({"Feature": feature_cols, "MI_Score": mi_scores}).sort_values("MI_Score", ascending=False).reset_index(drop=True)
    
    selected_features = mi_results[mi_results["MI_Score"] > threshold]["Feature"].tolist()
    
    excel_sel_path = os.path.join(result_dir, f"features_selected_{patient_name}.xlsx")
    excel_mi_path  = os.path.join(result_dir, f"mutual_info_{patient_name}.xlsx")
    
    df_selected = df[["Cell_Label", "X", "Y"] + selected_features + [target_col]]
    df_selected.to_excel(excel_sel_path, index=False)
    mi_results.to_excel(excel_mi_path, index=False)
    
    top5 = mi_results.head(5)["Feature"].tolist()
    return selected_features, mi_results, top5
