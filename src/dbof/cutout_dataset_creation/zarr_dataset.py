import zarr
import numpy as np
import uuid
import threading
from pathlib import PurePosixPath
import dask.array as da

def make_run_prefix(bucket: str, folder: str, run_id: str, dataset_name: str) -> str:
    bucket = bucket.strip().strip("/")
    folder = folder.strip().strip("/")
    return f"s3://{str(PurePosixPath(bucket, folder, run_id, dataset_name))}"

# Multi Thread Safe NOT Multi Process safe
class ZarrDataset:
    def __init__(self, bucket, folder, run_id, dataset_name, fs, channel_names, down_sample_res, target_km_res):
        path = make_run_prefix(bucket, folder, run_id, dataset_name)
        self.store = zarr.storage.FsspecStore(path=path, fs=fs)
        self.root = zarr.open_group(store=self.store, mode="a")

        C, H, W = len(channel_names), down_sample_res, down_sample_res
        self.C = C
        self.H = H
        self.W = W

        # channel_names is the stacking order of the image channels (channel c is
        # channel_names[c]); readers use it to map channels back to field names.
        self.root.attrs["channel_names"] = list(channel_names)
        self.root.attrs["target_km_res"] = target_km_res
        self.root.attrs["down_sample_res"] = down_sample_res

        if "images" not in self.root:
            self.root.create_array(
                "images",
                shape=(0, C, H, W),
                chunks=(1, C, H, W),
                dtype="float32"
            )

        # array to store image IDs
        if "image_ids" not in self.root:
            self.root.create_array(
                "image_ids",
                shape=(0,),
                chunks=(1,), # 1 string to protect overwrite in parallel writing
                dtype="S32" #"U32" # fixed len string for uuid
            )

        # set starting index. Will be 0 if new array or len of array if appending to an existing array
        # this style is required to link the metadata and zarr store
        self.zarr_global_index = self.root["images"].shape[0]

        self.lock = threading.Lock()

    # (Grow, C, H, W)
    def grow_array(self, len_to_grow):
        with self.lock:
            current_len = self.root["images"].shape[0]
            new_len = current_len + len_to_grow
            self.root["images"].resize((new_len, self.C, self.H, self.W))
            self.root["image_ids"].resize((new_len,) )


    # todo we should probably batch these before uploading
    def append_image(self, img):

        with self.lock: # lock index management to ensure safety across threads
            i = self.zarr_global_index
            self.zarr_global_index += 1

            if i >= self.root["images"].shape[0]:
                raise RuntimeError("ZarrDataset capacity exceeded; call grow_array first")

        image_id = uuid.uuid4().hex.encode("ascii")
        self.root["images"][i] = img.numpy()
        self.root["image_ids"][i] = image_id

        return image_id


class ZarrDatasetReader:
    """
    This class is useful for reading in data from remote zarr array.
    It has two main use cases
    Simple analysis -> viewing a few images at a time
    Training -> take the dask array of only valid images and process locally
    """

    def __init__(self, bucket, folder, run_id, dataset_name, fs):
        path = make_run_prefix(bucket, folder, run_id, dataset_name)

        self.store = zarr.storage.FsspecStore(path=path, fs=fs)
        self.root = zarr.open_group(store=self.store, mode="r")

        self.images = self.root["images"]
        self.image_ids = self.root["image_ids"]

        self.channel_names = (list(self.root.attrs["channel_names"])
                              if "channel_names" in self.root.attrs else None)
        self.target_km_res = self.root.attrs.get("target_km_res")
        self.down_sample_res = self.root.attrs.get("down_sample_res")

        # cached reverse index (lazy)
        self._id_to_index = None

    # --------------------------
    # Basic properties
    # --------------------------

    @property
    def shape(self):
        return self.images.shape

    @property
    def num_images(self):
        return self.images.shape[0]

    @property
    def channels(self):
        return self.images.shape[1]

    @property
    def height(self):
        return self.images.shape[2]

    @property
    def width(self):
        return self.images.shape[3]


    # --------------------------
    # Physical access
    # PLEASE NOTE : there can be holes in the zarr data due to failures in the data gen run.
    # These are not handled in any way by this code.
    # --------------------------

    def get_image(self, index):
        """
        Read a single image by integer index.

        Returns
        -------
        img : np.ndarray, shape (C, H, W)
        image_id : str
        """
        img = self.images[index]
        image_id = self.image_ids[index]

        return img, image_id

    def get_images(self, indices):
        """
        Read multiple images by indices.

        Parameters
        ----------
        indices : array-like of int

        Returns
        -------
        images : np.ndarray, shape (N, C, H, W)
        image_ids : list[str]
        """
        imgs = self.images[indices]
        ids = self.image_ids[indices].tolist()

        return imgs, ids

    def get_slice(self, start, stop):
        """
        Read a contiguous slice of images.

        Returns
        -------
        images : np.ndarray
        image_ids : list[str]
        """
        imgs = self.images[start:stop]
        ids = self.image_ids[start:stop].tolist()

        return imgs, ids

    # --------------------------
    # ID access
    # --------------------------

    def _build_id_index(self):
        """
        Build a mapping from image_id -> index.
        Done lazily and cached.
        Warning this will be super slow
        """
        id_to_index = {}
        for i, img_id in enumerate(self.image_ids[:]):
            id_to_index[img_id] = i
        return id_to_index

    def get_by_id(self, image_id):
        """
        Retrieve image by UUID string.

        Parameters
        ----------
        image_id : str

        Returns
        -------
        img : np.ndarray, shape (C, H, W)
        """
        if self._id_to_index is None:
            self._id_to_index = self._build_id_index()

        idx = self._id_to_index[image_id]
        img = self.images[idx]

        return img

    # --------------------------
    # Dask Access
    # this will be the most useful way to access this data for training
    # downstream dataloaders can load in data in parallel
    # --------------------------

    def full_dataset_as_dask(self):
        """
        This method returns the data as lazy dask arrays.
        User can decide how to mask invalid data given the mask and how to load data locally.

        Returns
        -------
        valid_images_da : dask.array.array
        valid_ids_da : dask.array.array
        valid_mask_da : dask.array.array
        """
        # dask arrays from zarr (lazy)
        images_da = da.from_zarr(self.images)  # (N, C, H, W)
        image_ids_da = da.from_zarr(self.image_ids)  # (N,)

        valid_mask_da = image_ids_da != b""

        return images_da.rechunk((1024, -1, -1, -1)), image_ids_da, valid_mask_da
