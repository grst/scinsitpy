import anndata as an
import scanpy as sc
import pandas as pd
import numpy as np
import cv2
import tifffile
import seaborn as sns
from matplotlib import pyplot as plt

def basic_plot(adata: an.AnnData) -> int:
    """Generate a basic plot for an AnnData object.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.

    Returns
    -------
    Some integer value.
    """
    print("Import matplotlib and implement a plotting function here.")
    return 0

def load_merscope(folder, library_id, scale_percent) -> an.AnnData:
    """Load vizgen merscope data.

    Parameters
    ----------
    folder
        path to folder.
    library_id
        library id.
    scale_percent
        scaling factor for image and pixel coordinates reduction.

    Returns
    -------
    Anndata initialized object.
    """

    # transformation matrix micron to mosaic pixel
    transformation_matrix = pd.read_csv(folder + "/images/micron_to_mosaic_pixel_transform.csv", header=None, sep=' ').values
    # genes
    data = pd.read_csv(folder + '/cell_by_gene.csv', index_col=0)
    datanoblank = data.drop(data.filter(regex='^Blank-').columns, axis=1)
    meta_gene = pd.DataFrame(index=datanoblank.columns.tolist())
    meta_gene['expression'] = datanoblank.sum(axis=0)
    # cells
    meta = pd.read_csv(folder + '/cell_metadata.csv', index_col=0)
    meta = meta.loc[data.index.tolist()]
    meta['cell_cov'] = datanoblank.sum(axis=1)
    meta['barcodeCount'] = datanoblank.sum(axis=1)
    meta["width"] = meta["max_x"].to_numpy() - meta["min_x"].to_numpy()
    meta["height"] = meta["max_y"].to_numpy() - meta["min_y"].to_numpy()
    
    # Transform coordinates to mosaic pixel coordinates
    temp = meta[['center_x', 'center_y']].values
    cell_positions = np.ones((temp.shape[0], temp.shape[1]+1))
    cell_positions[:, :-1] = temp
    transformed_positions = np.matmul(transformation_matrix, np.transpose(cell_positions))[:-1]
    meta['x_pix'] = transformed_positions[0, :] * (scale_percent/100)
    meta['y_pix'] = transformed_positions[1, :] * (scale_percent/100)
    meta['library_id'] = library_id
    meta = meta.drop(columns=['min_x', 'max_x', 'min_y', 'max_y'])
    
    # init annData pixel coordinates
    coord_pix = meta[['x_pix', 'y_pix']].to_numpy()
    coord_mic = meta[['center_x', 'center_y']].to_numpy()
    #coordinates.rename(columns={"x_pix": "x", "y_pix": "y"})
    adata = sc.AnnData(X=datanoblank.values, obsm={"spatial": coord_mic, "X_spatial": coord_mic, "pixel": coord_pix}, obs=meta, var=meta_gene)
    adata.layers["counts"] = adata.X.copy()
    
    # transcripts
    transcripts = load_transcript(folder + "/detected_transcripts.csv", transformation_matrix, scale_percent)
    adata.uns["transcripts"] = {library_id: {}}
    adata.uns["transcripts"][library_id] = transcripts
    
    percent_in_cell = adata.obs.barcodeCount.sum(axis=0)*100/adata.uns["transcripts"][library_id].shape[0]
    print('\n' + library_id)
    print('total cells=', adata.shape[0])
    print('total transcripts=', transcripts.shape[0])
    print('% in cells=', percent_in_cell)
    print('mean transcripts per cell=', meta['barcodeCount'].mean())
    print('median transcripts per cell=', meta['barcodeCount'].median())
    
    # image Container
    image = tifffile.imread(folder + "/images/mosaic_DAPI_z2.tif")
    #print('loaded image into memory')
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    #print('resized image')
    adata.uns["spatial"] = {library_id: {}}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"] = {"hires": resized}
    adata.uns["spatial"][library_id]["scalefactors"] = {"tissue_hires_scalef": 1, 
                                                        "spot_diameter_fullres": 5, 
                                                        "scale_percent": scale_percent, 
                                                        "transformation_matrix": transformation_matrix,
                                                        "folder": folder}
    image = None
    resized = None
    
    return adata

def filter_and_run_scanpy(adata, min_counts) -> an.AnnData:
    
    sc.pp.filter_cells(adata, min_counts = min_counts, inplace = True)
    
    print('total cells=', adata.shape[0])
    print('mean transcripts per cell=', adata.obs['barcodeCount'].mean())
    print('median transcripts per cell=', adata.obs['barcodeCount'].median())

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5, key_added = "clusters")
    
    fig, axs = plt.subplots(1, 2, figsize=(20,6))
    sc.pl.embedding(adata, "umap", color = "clusters", ax=axs[0], show = False)
    sns.scatterplot(x='x_pix', y='y_pix', data=adata.obs, s=2, hue = "clusters")
    
    return adata

def save_merscope(adata, file) -> int:
    adata.obs = adata.obs.drop(['bounds'], axis=1)
    #del adata.uns['transcripts']
    adata.write(file)
    return 0

def getFovCoordinates(fov, meta_cell) -> int():
    xmin = meta_cell.x[meta_cell.fov == fov].min()
    ymin = meta_cell.y[meta_cell.fov == fov].min()
    xmax = meta_cell.x[meta_cell.fov == fov].max()
    ymax = meta_cell.y[meta_cell.fov == fov].max()
    
    return (xmin,ymin,xmax,ymax)

def load_transcript(path, transformation_matrix, scale_percent) -> pd.DataFrame:
    transcripts = pd.read_csv(path, index_col=0)

    # Transform coordinates to mosaic pixel coordinates
    temp = transcripts[['global_x', 'global_y']].values
    transcript_positions = np.ones((temp.shape[0], temp.shape[1]+1))
    transcript_positions[:, :-1] = temp
    transformed_positions = np.matmul(transformation_matrix, np.transpose(transcript_positions))[:-1]
    transcripts['x_pix'] = transformed_positions[0, :] * (scale_percent/100)
    transcripts['y_pix'] = transformed_positions[1, :] * (scale_percent/100)
    
    # global_x, _y -> micron coordinates
    # x_pix, y_pix -> pixels coordinates
    return transcripts
