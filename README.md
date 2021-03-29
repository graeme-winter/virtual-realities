# virtual-realities
Scripts to support the virtual realities policy paper:

- `repacker.py` to repack the data with standard compression options
- `vds_internal.py` for internal VDS
- `vds_external.py` for external VDS

Usage:

```
for filename in $(cd ..; ls protk_1_13_3*000*); do
echo $filename;
dials.python ~/vds_internal.py ../$filename $filename;
done
```

This operates only on the _raw_ data files - the `master` file or
equivalent will need to be updated to have a top level virtual data
set if this is not already present in the data. In the examples used
in the paper this script was used:

```python
import h5py

with h5py.File("protk_1_13_3.nxs", "a") as f:
    d = f["/entry/data"]
    # 28,800 images @ 1000 / file, in protk_1_13_3_%06d.h5

    slow, fast = 4362, 4148

    layout = h5py.VirtualLayout(shape=(28800, slow, fast), dtype="i4")

    start = 0
    end = 1000

    while end < 28800:
        source = h5py.VirtualSource("protk_1_13_3_%06d.h5" % (end // 1000),
                                    "data", shape=(1000, slow, fast))
        layout[start:end, :, :] = source
        end += 1000
        start += 1000
    source = h5py.VirtualSource("protk_1_13_3_%06d.h5" % 29,
                                "data", shape=(800, slow, fast))
    layout[28000:28800,:,:] = source
    data = d.create_virtual_dataset("data", layout, fillvalue=-1)
```
