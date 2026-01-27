import zarr
import numpy as np
import uuid
import threading
import torch
from torch.utils.data import Dataset

# Multi Thread Safe not Multi Process safe
class ZarrDataset:
    def __init__(self, bucket, folder, run_id, dataset_name, fs, feature_channels, down_sample_res):
        self.store = zarr.storage.FsspecStore(path=bucket+folder+run_id+dataset_name, fs=fs)
        self.root = zarr.open_group(store=self.store, mode="a")

        C, H, W = len(feature_channels), down_sample_res, down_sample_res
        self.C = C
        self.H = H
        self.W = W

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

        with self.lock: # lock index management to ensure safety across threads or processes
            i = self.zarr_global_index
            self.zarr_global_index += 1

            if i >= self.root["images"].shape[0]:
                raise RuntimeError("ZarrDataset capacity exceeded; call grow_array first")

        image_id = uuid.uuid4().hex.encode("ascii") #uuid.uuid4().hex
        self.root["images"][i] = np.expand_dims(img, axis=0)
        self.root["image_ids"][i] = image_id

        return image_id


class ZarrDatasetReader:
    """
    Thread-safe reader for ZarrDataset written by ZarrDataset.

    Read-only. Safe for concurrent access from multiple threads.
    """

    def __init__(self, bucket, folder, run_id, dataset_name, fs):
        self.store = zarr.storage.FsspecStore(
            path=bucket + folder + run_id + dataset_name,
            fs=fs
        )
        self.root = zarr.open_group(store=self.store, mode="r")

        self.images = self.root["images"]
        self.image_ids = self.root["image_ids"]

        self.lock = threading.Lock()

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
    # Core accessors
    # --------------------------

    def get_image(self, index):
        """
        Read a single image by integer index.

        Returns
        -------
        img : np.ndarray, shape (C, H, W)
        image_id : str
        """
        with self.lock:
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
        with self.lock:
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
        with self.lock:
            imgs = self.images[start:stop]
            ids = self.image_ids[start:stop].tolist()

        return imgs, ids

    # --------------------------
    # ID-based access
    # --------------------------

    def _build_id_index(self):
        """
        Build a mapping from image_id -> index.
        Done lazily and cached.
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
        with self.lock:
            if self._id_to_index is None:
                self._id_to_index = self._build_id_index()

            idx = self._id_to_index[image_id]
            img = self.images[idx]

        return img

    # --------------------------
    # Iteration
    # --------------------------

    def iter_images(self, batch_size=1):
        """
        Iterate over dataset in batches.

        Yields
        ------
        images : np.ndarray, shape (B, C, H, W)
        image_ids : list[str]
        """
        n = self.num_images
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            yield self.get_slice(start, stop)


class ZarrTorchDataset(Dataset):
    """
    PyTorch Dataset wrapping ZarrDatasetReader.

    Each worker process gets its own reader instance.
    """

    def __init__(self, bucket, folder, run_id, dataset_name, fs, transform=None):
        self.bucket = bucket
        self.folder = folder
        self.run_id = run_id
        self.dataset_name = dataset_name
        self.fs = fs
        self.transform = transform

        # opened lazily per worker
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            self._reader = ZarrDatasetReader(
                bucket=self.bucket,
                folder=self.folder,
                run_id=self.run_id,
                dataset_name=self.dataset_name,
                fs=self.fs
            )
        return self._reader

    def __len__(self):
        reader = self._get_reader()
        return reader.num_images

    def __getitem__(self, idx):
        reader = self._get_reader()
        img, image_id = reader.get_image(idx)

        # (C, H, W) -> torch tensor
        img = torch.from_numpy(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, image_id
