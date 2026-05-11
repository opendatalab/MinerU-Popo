# Popo

This repository's post-processing pipeline relies on two types of references:
1. Data format reference: https://github.com/opendatalab/mineru.
   The goal is to first obtain OCR / layout parsing result formats compatible with MinerU, serving as input to `post_processing/inference.py`.

The main internal data flow of this repository is:
- `inference.py`: takes OCR result as input, outputs block-level JSON.
- `get_json_tree.py`: reads inference output, performs cross-page table merging, and builds tree JSON.
- `generate_metadata.py`: supplements title, metadata, and content on the tree JSON.
- `split_subnode.py`: further splits long text nodes to generate the final result.
