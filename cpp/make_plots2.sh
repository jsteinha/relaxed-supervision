#!/bin/bash
mkdir -p figures
python make_plots.py plots7.list
cp P-145_delta-0.10.pdf figures/predicate.pdf
python make_plots2.py plots6.list
cp S-50_delta-0.1_N-50.pdf figures/computation.pdf
cd figures
pdfcrop predicate.pdf
pdfcrop computation.pdf 
