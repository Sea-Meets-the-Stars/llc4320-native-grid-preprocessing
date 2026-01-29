import matplotlib.pyplot as plt
import cmocean
import numpy as np
# import dask.array as da

def plot_log_faces_layout(da, color_map = cmocean.cm.thermal, vmin=None, vmax=None):
    vals = np.abs(da.values.ravel())
    vals = vals[(vals > 0) & ~np.isnan(vals)]
    logvals = np.log10(vals)

    # find vmin and max for plot
    vmin = np.quantile(logvals, 0.01)
    vmax = np.quantile(logvals, 0.99)

    # create the log values xarray for plotting
    vals = np.abs(da)
    vals = vals.where(vals > 0)

    logvals_xarray = np.log10(vals)

    plot_llc_faces_layout(logvals_xarray, color_map, vmin=vmin, vmax=vmax)

def plot_pdf_dask(da_xr, title, bins=200):
    vals = np.abs(da_xr.values.ravel())
    vals = vals[(vals > 0) & ~np.isnan(vals)]

    plt.hist(vals, bins=200, density=True)
    plt.xlabel("value")
    plt.ylabel("PDF")
    plt.title(title)
    plt.show()


    # # da_xr: xarray with dask-backed data (face,i,j)
    # data = da_xr.data  # get dask array
    #
    # # Compute min and max lazily
    # vmin, vmax = da.min(data).compute(), da.max(data).compute()
    #
    # # Compute histogram lazily
    # hist, bin_edges = da.histogram(data, bins=bins, range=(vmin, vmax), density=True)
    #
    # # Trigger computation
    # hist_vals = hist.compute()
    # bin_edges_vals = bin_edges.compute()
    #
    # # Plot PDF
    # plt.figure()
    # plt.bar((bin_edges_vals[:-1] + bin_edges_vals[1:]) / 2, hist_vals, width=np.diff(bin_edges_vals), align='center')
    # plt.xlabel("Value")
    # plt.ylabel("PDF")
    # plt.title(title)
    # plt.show()

def plot_log_pdf(da, title):
    vals = np.abs(da.values.ravel())
    vals = vals[(vals > 0) & ~np.isnan(vals)]
    logvals = np.log10(vals)

    plt.hist(logvals, bins=200, density=True)
    plt.xlabel("log10(|value|)")
    plt.ylabel("PDF")
    plt.title(title)
    plt.show()


def plot_a_face_by_var(ds, face, color_map=cmocean.cm.thermal, vmin=None, vmax=None):
    # make a ds of this variable
    var = ds[face]

    print(var.dims, var.shape)

    var_slice = var
    print(var_slice.dims, var_slice.shape)
    print("Plotting slice with dims:", var_slice.dims)

    plt.figure(figsize=(20,12)) #cmap=cmocean.cm.thermal

    #var_slice.plot() #(vmin=-5.0, vmax=5.0)
    var_slice.plot.pcolormesh(cmap=color_map, vmin=vmin,vmax=vmax)

    #plt.title(f"{VARIABLE}")
    plt.show()

# plot_a_face_by_var(ds_merge.Theta, 0)

def plot_a_face_by_two_var(ds0,ds1, ds0_name, ds1_name, face,color_map=cmocean.cm.thermal, vmin0=None, vmax0=None, vmin1=None, vmax1=None):

    var0=ds0[face]
    var1=ds1[face]

    fig, axes = plt.subplots(1,2, figsize=(20,12))

    im0 = var0.plot.pcolormesh(ax=axes[0], cmap=color_map, vmin=vmin0,vmax=vmax0)
    axes[0].set_title(ds0_name)

    im1 = var1.plot.pcolormesh(ax=axes[1], cmap=color_map, vmin=vmin1,vmax=vmax1)
    axes[1].set_title(ds1_name)

    plt.show()

# todo you we need to generate vmin vmax or it will not be consistent accross faces
def plot_llc_faces_layout(ds, color_map = cmocean.cm.thermal, vmin=None, vmax=None, is_mask=False):
    # Layout (row, col) positions of the 13 faces
    layout = {
        0:  (4, 0),
        1:  (3, 0),
        2:  (2, 0),
        3:  (4, 1),
        4:  (3, 1),
        5:  (2, 1),
        6:  (1, 1),
        7:  (1, 2),
        8:  (1, 3),
        9:  (1, 4),
        10: (0, 2),
        11: (0, 3),
        12: (0, 4),
    }

    if vmin is None:
        # find vmin and max for plot
        vmin = ds.min().compute()
        vmax = ds.max().compute()

    fig, axes = plt.subplots(5, 5, figsize=(14, 14))
    axes = axes.flatten()

    # hide all axes first
    for ax in axes:
        ax.axis("off")

    mappable = None  # to store color mapping for colorbar

    for face, (row, col) in layout.items():
        var = ds[face]
        ax = axes[row * 5 + col]

        if not is_mask:
            mappable= var.plot(ax=ax, add_colorbar=False, cmap=color_map, vmin=vmin, vmax=vmax)
        else:
            mappable = var.plot(ax=ax, add_colorbar=False, cmap=color_map, vmin=vmin, vmax=vmax, shading="nearest", infer_intervals=False)

        ax.set_title(f"Face {face}", fontsize=10)
        ax.axis("off")

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, orientation="horizontal", fraction=0.05, pad=0.05)
        cbar.set_label(ds.name if hasattr(ds, "name") else "")

    plt.tight_layout()
    plt.show()



