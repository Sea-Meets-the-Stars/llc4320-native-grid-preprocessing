import zarr
import numpy as np
import uuid
import threading

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
                chunks=(1024,),
                dtype="U32" # fixed len string for uuid
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

        image_id = uuid.uuid4().hex
        self.root["images"][i] = np.expand_dims(img, axis=0)
        self.root["image_ids"][i] = image_id

        return image_id
