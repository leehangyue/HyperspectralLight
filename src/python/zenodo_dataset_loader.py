import pandas as pd
import numpy as np
from os.path import join, dirname

"""
Source: https://zenodo.org/records/5217752
"""

class ZenodoDataset:
    FNAMES = {
        # "daylight_timelapse": "Daylight_TimeLapse_v1.xlsx",
        "daylight_timelapse": "Daylight_TimeLapse_v1-2.xlsx",
        "daylight_locations": "Daylight_DifferentLocations_v1-2.xlsx",
        # "reflectance": "Reflectance.xlsx",
        # "reflectance": "Reflectance_v1-1.xlsx",
        "reflectance": "Reflectance_v1-2.xlsx",
        # "transmittance_front": "Transmittance_FrontSideUp.csv",
        # "transmittance_front": "Transmittance_FrontSideUp_v1-1.xlsx",
        "transmittance_front": "Transmittance_FrontSideUp_v1-2.xlsx",
        # "transmittance_back": "Transmittance_BackSideUp.xlsx",
        # "transmittance_back": "Transmittance_BackSideUp_v1-1.xlsx",
        "transmittance_back": "Transmittance_BackSideUp_v1-2.xlsx",
    }
    DIRNAME = join(dirname(dirname(dirname(__file__))), "res", "zenodo_dataset")
    
    def __init__(self):
        self.data = dict()
    
    def load(self, label):
        if label in self.data.keys():
            return self.data[label]
        if label not in self.FNAMES.keys():
            raise ValueError(f"label {label} not recognized. Must be one of: {list(self.FNAMES.keys())}")
        
        fname = self.FNAMES[label]
        skiprows = 1 if "daylight" in fname.lower() else 0
        if fname.lower().endswith(".csv"):
            df = pd.read_csv(join(self.DIRNAME, fname), skiprows=skiprows)
        elif fname.lower().endswith(".xlsx"):
            df = pd.read_excel(join(self.DIRNAME, fname), skiprows=skiprows)
        else:
            raise NotImplementedError(f"Unexpected extension: {fname}, expecting .csv or .xlsx")
        cols = df.columns.tolist()
        idx_col_spectrum_start = [isinstance(col, str) for col in cols].index(False) - 1
        n_datarows = df.shape[0]
        row_dict_list = []
        for idx_row in range(n_datarows):
            row_dict = dict()
            for idx_col, col in enumerate(cols[:idx_col_spectrum_start]):
                row_dict[col] = df.iloc[idx_row, idx_col]
            if "wavelengths" not in row_dict.keys():
                wavelengths = cols[idx_col_spectrum_start:]
                for idx_wl, wl in enumerate(wavelengths):
                    if isinstance(wl, str):
                        wl = float(wl.strip("[nm]"))
                    wavelengths[idx_wl] = wl
                row_dict["wavelengths"] = np.array(wavelengths, dtype=float)
            row_dict["spectrum"] = df.iloc[idx_row, idx_col_spectrum_start:].to_numpy(dtype=float)
            row_dict_list.append(row_dict)
        self.data[label] = row_dict_list
        return row_dict_list

    def load_all(self):
        for label in self.FNAMES.keys():
            self.load(label=label)
        return self.data


def test_zenodo_dataset():
    zenodo_dataset = ZenodoDataset()
    zenodo_dataset.load_all()
    import matplotlib.pyplot as plt
    plt.plot(
        zenodo_dataset.data["daylight_timelapse"][0]["wavelengths"],
        zenodo_dataset.data["daylight_timelapse"][0]["spectrum"]
    )
    plt.show()
    print("Test completed")


if __name__ == "__main__":
    test_zenodo_dataset()

