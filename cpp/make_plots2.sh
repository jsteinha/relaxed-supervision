#!/bin/bash
mkdir -p figures
mkdir -p figures/accuracy
mkdir -p figures/computation
python make_plots.py plots7-$1.list
cp P-145_delta-0.10.pdf figures/accuracy/predicate-$1.pdf
python make_plots2.py plots7-$1.list
cp P-145_delta-0.10.pdf figures/computation/predicate-$1.pdf
#python make_plots2.py plots6.list
#cp S-50_delta-0.1_N-50.pdf figures/computation.pdf
#python make_plots2.py plots6b.list
#cp S-50_delta-0.1_r-1.2_N-100.pdf figures/powerlaw_computation.pdf
cd figures/accuracy
pdfcrop predicate-$1.pdf
cd ../computation
pdfcrop predicate-$1.pdf
#pdfcrop predicate_computation.pdf
#pdfcrop powerlaw_computation.pdf
#pdfcrop computation.pdf 
