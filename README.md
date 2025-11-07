# quicktree2histo
What this does
- quicktree2histo.py reads a YAML configuration (hists.yaml), opens input ROOT TTree(s), optionally attaches friend trees, builds histograms with ROOT RDataFrame and writes them into an output ROOT file.
- It includes debug prints to inspect TTree/RDataFrame columns and sample values when histograms are empty.

How to run
1. Prepare a YAML config (example: plot-recoil/hists.yaml).
2. Run:
   python3 quicktree2histo.py plot-recoil/hists.yaml
