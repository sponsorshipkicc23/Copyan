import os
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Session:
    patient_name: str = "Anonim"
    base_dir: str = ""
  
    session_dir: str = field(default="", init=False)
    raw_dir: str = field(default="", init=False)
    clust_dir: str = field(default="", init=False)
    sep_dir: str = field(default="", init=False)
    res_dir: str = field(default="", init=False)

    def create(self, master_data_dir: str) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(master_data_dir,
                                        f"{self.patient_name}_{timestamp}")
        self.raw_dir   = os.path.join(self.session_dir, "0_raw_image")
        self.clust_dir = os.path.join(self.session_dir, "1_clustering_image")
        self.sep_dir   = os.path.join(self.session_dir, "2_separated_cells")
        self.res_dir   = os.path.join(self.session_dir, "3_results")

        for folder in (self.raw_dir, self.clust_dir, self.sep_dir, self.res_dir):
            os.makedirs(folder, exist_ok=True)

    def raw_image_path(self) -> str:
        return os.path.join(self.raw_dir, f"raw_{self.patient_name}.jpg")

    def cluster_image_path(self, cluster_idx: int) -> str:
        return os.path.join(self.clust_dir, f"cluster_{cluster_idx + 1}.jpg")

    def detect_initial_path(self) -> str:
        return os.path.join(self.res_dir, "detect_cells_initial.jpg")

    def after_sep_path(self) -> str:
        return os.path.join(self.res_dir, "after_sep.jpg")

    def features_path(self) -> str:
        return os.path.join(self.res_dir, f"features_{self.patient_name}.xlsx")

    def features_selected_path(self) -> str:
        return os.path.join(self.res_dir,
                            f"features_selected_{self.patient_name}.xlsx")

    def mutual_info_path(self) -> str:
        return os.path.join(self.res_dir,
                            f"mutual_info_{self.patient_name}.xlsx")

    def detect_result_path(self) -> str:
        return os.path.join(self.res_dir,
                            f"detect_result_{self.patient_name}.png")

    def pdf_report_path(self) -> str:
        return os.path.join(self.res_dir,
                            f"Report_{self.patient_name}.pdf")


def new_session(patient_name: str, master_data_dir: str) -> Session:
    clean_name = (patient_name.strip().replace(" ", "_") or "Anonim")
    session = Session(patient_name=clean_name, base_dir=master_data_dir)
    session.create(master_data_dir)
    return session
