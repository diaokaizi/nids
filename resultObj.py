import matplotlib.pyplot as plt
from datetime import datetime
import os
import pandas as pd
class Result:
    def __init__(self):
        self.time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = os.path.join("/root/work/NIDS/training_results", self.time)
        os.makedirs(self.base_dir, exist_ok=True)

    def draw_loss(self, history, filename):
        plt.figure()
        plt.plot(history.history['loss'], label="Training loss")
        plt.plot(history.history['val_loss'], label="Validation loss", ls="--")
        plt.legend(shadow=True, frameon=True, facecolor="inherit", loc="best", fontsize=9)
        plt.title("Training loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        file_path = self.get_pic_path(filename)
        plt.savefig(file_path)
        plt.close()
    
    def get_pic_path(self, filename):
        return os.path.join(self.base_dir, f"{filename}.png")
    def get_xlsx_path(self):
        return os.path.join(self.base_dir, "result.xlsx")
    
    def save_result(self, roc_metrics, pr_metrics, filename):
        df_roc = pd.DataFrame(roc_metrics, index=['ROC'])
        df_pr = pd.DataFrame(pr_metrics, index=['PR'])
        df_result = pd.concat([df_roc, df_pr])
        excel_path = self.get_xlsx_path()
        if os.path.exists(excel_path):
            with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='replace') as writer:
                df_result.to_excel(writer, sheet_name=filename)
        else:
            with pd.ExcelWriter(excel_path, mode='w') as writer:
                df_result.to_excel(writer, sheet_name=filename)
        print(f"Evaluation results saved to {excel_path}")