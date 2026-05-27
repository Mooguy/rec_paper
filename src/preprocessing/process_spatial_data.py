import pandas as pd

from src.utils.utils import find_project_root

BASE_DIR = find_project_root()
EXTERNAL_DATA_DIR = BASE_DIR / "data" / "external"

# Output from MATLAB script - compute_spatial_coordinates_zones_and_crypt.m

cell_zone_table_with_crypt = pd.read_csv(
    EXTERNAL_DATA_DIR / "cell_spatial_coordinates_and_zones.csv", sep=","
).rename(columns={"Barcode": "barcode", "Spatial_Coordinate": "spatial_coordinate", "Zone": "zone"})

seurat_data = pd.read_csv(
    EXTERNAL_DATA_DIR / "CSC_dataset_seurat_tsne_export_171212.txt", sep=" "
).rename(columns={"rn": "barcode", "tSNE_1": "tsne_1", "tSNE_2": "tsne_2", "entcrypt.subset.filt.ident": "enteroctyte_identity"})


transient_cells = seurat_data[seurat_data['enteroctyte_identity'].isin(['transient_amp_2','transient_amp_1'])]['barcode'].values

# Set zone to 0 for transient cells
cell_zone_table_with_crypt.loc[cell_zone_table_with_crypt['barcode'].isin(transient_cells), 'zone'] = 0

#Save as cell_zone_table_with_crypt.txt:
cell_zone_table_with_crypt.to_csv(EXTERNAL_DATA_DIR / "cell_zone_table_with_crypt.txt", sep="\t", index=False)
print("cell_zone_table_with_crypt.txt saved successfully.")