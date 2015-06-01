import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import fileinput
args = []
for line in fileinput.input():
  toks = line.rstrip("\n").split()
  arg = dict()
  for tok in toks:
    kv = tok.split(":")
    arg[kv[0]] = kv[1]
  args.append(arg)
print 'args', args
matches = [[] for _ in args]

outputdir = "/afs/cs.stanford.edu/u/jsteinhardt/NLP-HOME/scr/relaxed-supervision/output"
from os import listdir
from os.path import isfile, join
from itertools import islice
files = [ f for f in listdir(outputdir) if isfile(join(outputdir,f)) ]
for f in files:
  if f[0] == '.':
    continue
  head = list(islice(open(join(outputdir,f)), 100))
  arg = dict()
  for line in head:
    toks = line.rstrip("\n").split()
    if len(toks) > 0 and toks[0] == 'OPTION':
      arg[toks[1]] = toks[2]
  for ii, arg2 in enumerate(args):
    ok = True
    for k, v in arg2.items():
      if k in arg and arg[k] == v:
        pass
      else:
        try:
          ok = k in arg and abs(float(v)-float(arg[k])) < 1e-6
        except ValueError:
          ok = False
        if not ok:
          break
    if ok:
      matches[ii].append(f)

for arg, match in zip(args, matches):
  if len(match) != 1:
    raise Exception('non-unique match for %s: %s' % (arg, match))
  name = match[0]
  f = open(join(outputdir, match[0]))
  iterations = []
  prev_beta = False
  prev_theta_diag = False
  prev_freqs = False
  iteration = []
  for line in f:
    toks = line.rstrip("\n").split()
    if len(toks) == 0:
      continue
    if "iteration" in toks:
      if iteration:
        iterations.append(iteration)
      iteration = dict()
      if "(stage=1)" in toks:
        iteration['stage'] = 1
      elif "(stage=2)" in toks:
        iteration['stage'] = 2
      else:
        iteration['stage'] = 0
      iteration['samples'] = dict()
    if toks[0] == "SAMPLES":
      iteration['samples'][int(toks[1])] = float(toks[2])
    if "Average" in toks and "samples:" in toks:
      iteration['avg_samples'] = float(toks[-1])
    if toks[0] == "Trace:":
      iteration['trace'] = float(toks[-1])
    if toks[0] == "Trace2:":
      iteration['trace2'] = float(toks[-1])
    if prev_beta:
      iteration['beta'] = map(float, toks)
    if prev_theta_diag:
      iteration['theta'] = map(float, toks)
    prev_beta = "BETA:" in toks
    prev_theta_diag = "THETA_diag:" in toks
    prev_freqs = "FREQS:" in toks
  print iterations
  plt.clf()
  plt.hold(True)
  plt.plot([it['trace'] for it in iterations if 'trace' in it], 'b', label='trace')
  plt.plot([it['avg_samples'] for it in iterations if 'avg_samples' in it], 'k', label='samples')
  plt.savefig('%s.pdf' % name)
