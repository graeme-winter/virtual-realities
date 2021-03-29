import os
import shutil
import sys

import numpy as np
import h5py
import bitshuffle.h5


def repack(input_h5, output_h5):
    """Read the data file, create N_FAST * N_SLOW new data sets, then
    copy the data from the former into the latter and build a VDS"""

    with h5py.File(input_h5, "r") as fin, h5py.File(output_h5, "x") as fout:
        frames, slow, fast = fin["data"].shape

        dset = fout.create_dataset(
            "data",
            (frames, slow, fast),
            chunks=(1, slow, fast),
            compression=bitshuffle.h5.H5FILTER,
            compression_opts=(0, bitshuffle.h5.H5_COMPRESS_LZ4),
            dtype=fin["data"].dtype,
        )

        for j in range(frames):
            image = fin["data"][j, :, :]
            dset[j, :, :] = image

        for k in "image_nr_low", "image_nr_high":
            dset.attrs.create(k, fin["data"].attrs.get(k), dtype="i4")


if __name__ == "__main__":
    repack(sys.argv[1], sys.argv[2])
