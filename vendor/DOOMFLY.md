# Pinned Doomfly source

`doomfly/` contains byte-identical files from
https://github.com/nftechie/doomfly/tree/71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33.
`doomfly-source.json` records every copied source file's SHA-256. The selected
modules implement the original all-edge import, graph preparation, and native
CPU LIF simulation; they do not require ViZDoom or Brian2 at runtime. Preserve
the upstream license and third-party notices included alongside them.

The graph setup downloads all three official MaleCNS v1.0 feather files and
requires exact matches to the committed upstream source lock. Original feather
files (including anatomical fields/coordinates) are retained unchanged. The
upstream importer retains all 166,700 neuronal candidates and all 25,582,938
released directed connections between them: 124,177,617 synaptic contacts.
The graph is indexed exactly as upstream; no extra pruning, neuron merging, or
connectivity simplification is performed.

`doomfly-reference-manifest.json` is a byte-identical copy of upstream's
`outputs/doom/malecns_v1/manifest.json`, stored outside the generated output
directory. The setup checks regenerated counts and readout metadata against
that immutable reference.

The saved upstream reference lists its ten biological-role readouts first, then
appends four visual BCI readouts. Its pinned `prepare.py` emits the same fourteen
records in ascending neuron-index order. Setup verifies exact record equality
after sorting by neuron index; this difference does not change any neuron ID,
type, side, or graph edge. The discrepancy is recorded in the setup report.

On macm3:

```sh
cd /Users/fernando/fly_wordbrain
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-graph.txt
.venv/bin/python scripts/prepare_graph.py
```

Outputs live below `vendor/doomfly/connectome_data/malecns_v1/` and
`vendor/doomfly/outputs/doom/`. These large generated data and host-specific
native binaries must remain untracked. `results/graph-setup.json` records the
verified source/dataset/build information and exact normalized graph counts.
