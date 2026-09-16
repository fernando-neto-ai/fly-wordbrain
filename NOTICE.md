# Attribution and terms

The MIT licence in [LICENSE](LICENSE) covers **this repository's own source code,
configurations and documentation**. It does not, and cannot, relicense the third-party
material below. Each item keeps the terms it arrived with.

## Connectome

Neuron identities, connectivity, cell types and soma positions derive from the
**MaleCNS v1.0** connectome, released under **CC BY 4.0** by FlyEM / HHMI Janelia Research
Campus, the University of Cambridge, the MRC Laboratory of Molecular Biology and
Google Research. Any redistribution of graph-derived artifacts — including our published
model weights and packaged edge files — carries that attribution and that licence.

## Reference model

The architecture, tokenizer and graph packaging come from
[`ngxson/fly-llm-hf`](https://huggingface.co/ngxson/fly-llm-hf), revision
`65c677b3d566a2e9793d5f72999cdb441c6c0a9f`, whose model card declares **CC BY 4.0**.
Our trained weights are independently trained parameters over that released graph and
interface; they contain none of the released model's learned values.

## Libraries

[ConnecTorch](https://github.com/us/connectorch) is **MIT**; historical experiments pin
revision `4bbfb645099aeb85bdbf850e1a87cc094769af87`. Our Apple Metal backend contribution
to that project was supplied under ConnecTorch's MIT licence, which does not relicense
anything else here.

## Text corpus

Training, validation and audit stories are drawn from
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) by Eldan and Li,
licensed **CDLA-Sharing-1.0**. **We do not redistribute the corpus.** This repository and
our published artifacts carry only story identifiers, SHA-256 digests, counts and retrieval
receipts, so anyone can reconstruct the exact splits from the upstream dataset under its own
terms.

## Demo assets

The browser demo bundles a NeuroMechFly-derived body rig and fonts with their own notices,
retained under `demo/public/assets/` and `demo/public/fonts/`. See
[demo/REFERENCES.md](demo/REFERENCES.md). The body geometry is a **female** specimen; the
connectome is from a **male** specimen. They are different animals, and the overlay is
illustrative registration, not co-registered measurement.

## Vendored material

The vendored Doomfly subset retains its [MIT licence](vendor/doomfly/LICENSE),
[third-party notices](vendor/doomfly/THIRD_PARTY_NOTICES.md) and
[source inventory](vendor/DOOMFLY.md). Those notices reference the upstream repository's own
layout, which is deliberately not copied here in full.
