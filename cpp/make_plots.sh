#!/bin/bash
mkdir -p figures
mkdir -p figures/accuracy
mkdir -p figures/computation
python make_plots.py plots_all.list
cp S-50_N-50.pdf             figures/accuracy/noiseless.pdf
cp S-50_delta-0.1_N-50.pdf   figures/accuracy/iid_noise.pdf
cp N-250.pdf                 figures/accuracy/iid_noise_large.pdf
cp r-1.2_N-100.pdf           figures/accuracy/powerlaw_noiseless.pdf
cp delta-0.1_r-1.2_N-100.pdf figures/accuracy/powerlaw_iid_noise.pdf
python make_plots2.py plots_all.list
cp S-50_N-50.pdf             figures/computation/noiseless.pdf
cp S-50_delta-0.1_N-50.pdf   figures/computation/iid_noise.pdf
cp N-250.pdf                 figures/computation/iid_noise_large.pdf
cp r-1.2_N-100.pdf           figures/computation/powerlaw_noiseless.pdf
cp delta-0.1_r-1.2_N-100.pdf figures/computation/powerlaw_iid_noise.pdf

cd figures
cd accuracy
pdfcrop noiseless.pdf
pdfcrop iid_noise.pdf
pdfcrop iid_noise_large.pdf
pdfcrop powerlaw_noiseless.pdf
pdfcrop powerlaw_iid_noise.pdf
cd ../computation
pdfcrop noiseless.pdf
pdfcrop iid_noise.pdf
pdfcrop iid_noise_large.pdf
pdfcrop powerlaw_noiseless.pdf
pdfcrop powerlaw_iid_noise.pdf
