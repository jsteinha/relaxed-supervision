#!/bin/bash
./main -a 0 -N $1 -S $2 -T 50.0 -L 20 -d $3 -e $4 -r $5 -B $6 > output6/a0_t0_N$1_S$2_d$3_e$4_r$5_B$6.out &
./main -a 0 -t -N $1 -S $2 -T 50.0 -L 20 -d $3 -e $4 -r $5 -B $6 > output6/a0_t1_N$1_S$2_d$3_e$4_r$5_B$6.out &
