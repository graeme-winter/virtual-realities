# Eiger9M as Modules
#
# Take the 8 modules from an Eiger 2XE 9M and write out as 18 x 1-module data
# sets as 18 HDF5 stacks, using the standard commpression etc.

import sys

import numpy as np
import h5py
import bitshuffle.h5

MOD_FAST = 1028
MOD_SLOW = 512
GAP_FAST = 12
GAP_SLOW = 38
N_FAST = 3
N_SLOW = 6


def blit(source_data_set, sink_data_sets):
    """Blit the data from individual modules from the regions of the source
    images to N_FAST x N_SLOW sink data sets"""

    frames, slow, fast = source_data_set.shape

    assert fast == N_FAST * MOD_FAST + (N_FAST - 1) * GAP_FAST
    assert slow == N_SLOW * MOD_SLOW + (N_SLOW - 1) * GAP_SLOW

    for sink in sink_data_sets:
        assert sink.shape[0] == frames

    for k in range(frames):
        image = source_data_set[k, :, :]

        for j in range(N_SLOW):
            for i in range(N_FAST):
                f0 = i * (MOD_FAST + GAP_FAST)
                f1 = f0 + MOD_FAST
                s0 = j * (MOD_SLOW + GAP_SLOW)
                s1 = s0 + MOD_SLOW
                sink_data_sets[j * N_FAST + i][k, :, :] = image[s0:s1, f0:f1]


def split(input_h5, output_h5):
    """Read the data file, create N_FAST * N_SLOW new data sets, then
    copy the data from the former into the latter and build a VDS"""

    with h5py.File(input_h5, "r") as fin:
        frames, slow, fast = fin["data"].shape

        output_files = []
        output_dsets = []
        for n in range(N_SLOW * N_FAST):
            filename = output_h5.replace(".h5", "_%02d.h5" % n)
            fout = h5py.File(filename, "x")

            dset = fout.create_dataset(
                "data",
                (frames, MOD_SLOW, MOD_FAST),
                chunks=(1, MOD_SLOW, MOD_FAST),
                compression=bitshuffle.h5.H5FILTER,
                compression_opts=(0, bitshuffle.h5.H5_COMPRESS_LZ4),
                dtype=fin["data"].dtype,
            )

            output_files.append((fout, filename))
            output_dsets.append(dset)

        blit(fin["data"], output_dsets)

        for fout in output_files:
            fout[0].close()

        # create VDS
        layout = h5py.VirtualLayout(shape=(frames, slow, fast), dtype="i4")

        for j in range(N_SLOW):
            for i in range(N_FAST):

                source = h5py.VirtualSource(
                    output_files[j * N_FAST + i][1],
                    "data",
                    shape=(frames, MOD_SLOW, MOD_FAST),
                )
                f0 = i * (MOD_FAST + GAP_FAST)
                f1 = f0 + MOD_FAST
                s0 = j * (MOD_SLOW + GAP_SLOW)
                s1 = s0 + MOD_SLOW
                layout[:, s0:s1, f0:f1] = source[:, :, :]

        fout = h5py.File(output_h5, "x")
        data = fout.create_virtual_dataset("data", layout, fillvalue=-1)
        for k in "image_nr_low", "image_nr_high":
            data.attrs.create(k, fin["data"].attrs.get(k), dtype="i4")


if __name__ == "__main__":
    split(sys.argv[1], sys.argv[2])
