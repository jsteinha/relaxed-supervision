#!/bin/bash
mkdir -p figures
python make_plots.py plots_all.list
cp S-50_N-50.pdf figures/noiseless.pdf
cp S-50_delta-0.1_N-50.pdf figures/iid_noise.pdf
cp N-250.pdf figures/iid_noise_large.pdf
cp r-1.2_N-100.pdf figures/powerlaw_noiseless.pdf
cp delta-0.1_r-1.2_N-100.pdf figures/powerlaw_iid_noise.pdf
cd figures
pdfcrop noiseless.pdf
pdfcrop iid_noise.pdf
pdfcrop iid_noise_large.pdf
pdfcrop powerlaw_noiseless.pdf
pdfcrop powerlaw_iid_noise.pdf
