import zarr
import numpy as np
import uuid
import threading
import torch
from torch.utils.data import Dataset
from pathlib import PurePosixPath

# todo zarr and metadata can have empty spaces in the dataset. Shouldn't but can. We should account for this with our readers
def make_run_prefix(bucket: str, folder: str, run_id: str, dataset_name: str) -> str:
    bucket = bucket.strip().strip("/")
    folder = folder.strip().strip("/")
    return f"s3://{str(PurePosixPath(bucket, folder, run_id, dataset_name))}"

# Multi Thread Safe not Multi Process safe
class ZarrDataset:
    def __init__(self, bucket, folder, run_id, dataset_name, fs, num_channels, down_sample_res):
        path = make_run_prefix(bucket, folder, run_id, dataset_name)

        self.store = zarr.storage.FsspecStore(path=path, fs=fs)

        # self.store = zarr.storage.FsspecStore(path=bucket+folder+run_id+dataset_name, fs=fs)
        self.root = zarr.open_group(store=self.store, mode="a")

        C, H, W = num_channels, down_sample_res, down_sample_res
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

        image_id = uuid.uuid4().hex.encode("ascii")
        self.root["images"][i] = img.numpy() #np.expand_dims(img, axis=0) #todo I dont think we need to expand dims here, double check
        self.root["image_ids"][i] = image_id

        return image_id


class ZarrDatasetReader:
    """
    Thread-safe reader for ZarrDataset written by ZarrDataset.

    Read-only. Safe for concurrent access from multiple threads.
    """

    def __init__(self, bucket, folder, run_id, dataset_name, fs):
        path = make_run_prefix(bucket, folder, run_id, dataset_name)

        self.store = zarr.storage.FsspecStore(path=path, fs=fs)

        # --- valid physical indices cache (built once) ---
        self.valid_indices = None          # np.ndarray of physical indices
        self._iter_pos = 0                 # pointer for get_next()/iteration


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

    @property
    def num_valid_images(self):
        if self.valid_indices is None:
            self.build_valid_indices()
        return int(self.valid_indices.size)

    def reset_iterator(self):
        self._iter_pos = 0

    def build_valid_indices(self, chunk_size=100_000):
        """
        Build and cache an array of *physical* indices that have valid data.
        This is a one-time scan over image_ids; training then uses only these indices.
        """
        valid = []
        n = self.num_images

        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            with self.lock:
                ids = self.image_ids[start:stop]

            # normalize -> python strings (handles bytes/object)
            ids = np.asarray(ids).astype(str)

            mask = ids != "" # converted to str above so empty bytes are ""
            idxs = np.nonzero(mask)[0] + start
            if idxs.size:
                valid.append(idxs.astype(np.int64))

        self.valid_indices = np.concatenate(valid) if valid else np.array([], dtype=np.int64)
        self._iter_pos = 0
        return self.valid_indices

    # --------------------------
    # Core accessors
    # Physical ID-based access
    # PLEASE NOTE : there can be holes in the zarr data due to failures in the data gen run.
    # These are not handled in any way by this code.
    # However, the get next or acessing via metadata indexes are safe methods
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

    def get_next(self):
        """
        Return the next valid sample (skipping holes), using physical indices internally.
        Raises StopIteration when exhausted.
        """
        if self.valid_indices is None:
            self.build_valid_indices()

        if self._iter_pos >= self.valid_indices.size:
            raise StopIteration

        phys_idx = int(self.valid_indices[self._iter_pos])
        self._iter_pos += 1

        return self.get_image(phys_idx)

    def iter_images(self, batch_size=1):
        """
        Iterate over *valid* samples only, in batches.
        """
        if self.valid_indices is None:
            self.build_valid_indices()

        v = self.valid_indices
        for start in range(0, v.size, batch_size):
            phys = v[start:start + batch_size]
            yield self.get_images(phys)


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
