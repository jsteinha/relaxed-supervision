import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

#outputdir = "/afs/cs.stanford.edu/u/jsteinhardt/NLP-HOME/scr/relaxed-supervision/output"
#outputdir = "bvsN"
outputdir = "output5"
#outputdir = "output4"

import fileinput
args = []
#opts = ['trace', 'trace2', 'beta']
#opts = ['avg_samples']
opts = ['trace2']
agg_opts = [] #[('beta_val', 'tied_beta', 'N')]
agg = [defaultdict(list) for _ in agg_opts]
descs = defaultdict(dict)
for line in fileinput.input():
  toks = line.rstrip("\n").split()
  if toks[0] == 'dir':
    outputdir = toks[1]
    continue
  if toks[0] == 'desc':
    order = int(toks[1])
    name = toks[2]
    name2 = ' '.join(toks[3:])
    descs[id(cur_arg)][name] = (order, name2)
    continue
  arg = dict()
  for tok in toks:
    kv = tok.split(":")
    arg[kv[0]] = kv[1]
  args.append(arg)
  cur_arg = arg
print 'args', args
matches = [[] for _ in args]

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
      matches[ii].append((f, arg))

#markers = ['.', 'x', '^', 's', '+', 'o', '*', 'v']
colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y']
for arg, match in zip(args, matches):
  desc = '_'.join(['%s-%s' % (k,v) for k, v in arg.items() if float(v) != 0.0])
  if desc == '':
    desc = 'all'
  plt.figure(1)
  plt.clf()
  plt.hold(True)
  #if len(match) != 1:
  #  raise Exception('non-unique match for %s: %s' % (arg, match))
  handles_dict = dict()
  for ii, tup in enumerate(match):
    ma, full_arg = tup
    if id(arg) in descs and ma in descs[id(arg)]:
      name = descs[id(arg)][ma][1]
      order= descs[id(arg)][ma][0]
    else:
      name = ma
      order = ii
    if order == -1:
      continue
    f = open(join(outputdir, ma))
    iterations = []
    prev_beta = False
    prev_theta_diag = False
    prev_freqs = False
    iteration = []
    for line in f:
      toks = line.rstrip("\n").split()
      if len(toks) == 0:
        prev_beta = "BETA:" in toks
        prev_theta_diag = "THETA_diag:" in toks
        prev_freqs = "FREQS:" in toks
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
      if prev_freqs:
        iteration['freqs'] = map(int, toks)
      prev_beta = "BETA:" in toks
      prev_theta_diag = "THETA_diag:" in toks
      prev_freqs = "FREQS:" in toks
    print iterations
    if 'trace' in opts:
      V = full_arg['W']
      plt.plot([it['trace']/float(V) for it in iterations if 'trace' in it], '%s-o' % colors[ii % len(colors)]) #, label='trace (%s)' % name)
    if 'trace2' in opts:
      V = full_arg['W']
      line, = plt.plot([it['trace2']/float(V) for it in iterations if 'trace2' in it], '.-', color=colors[order % len(colors)]) #, label='%s' % name)
      handles_dict[order] = (line, name)
    if 'avg_samples' in opts:
      line, = plt.semilogy([it['avg_samples'] for it in iterations if 'avg_samples' in it], '%s-s' % colors[order % len(colors)], label='samples/100 (%s)' % name)
      handles_dict[order] = (line, name)
    if 'beta' in opts and 'a0' in name and 't0' in name:
      plt.figure(2)
      f, axarr = plt.subplots(2)
      axarr[0].matshow([it['beta'] for it in iterations if 'beta' in it])
      #plt.colorbar()
      plt.title(name)
      freqs = iterations[0]['freqs']
      axarr[1].bar(range(len(freqs)), freqs)
      plt.savefig('beta_%s.pdf' % name)
      plt.figure(1)
    for ii, opt in enumerate(agg_opts):
      accs = [it['trace2']/102.0 for it in iterations if 'trace2' in it]
      if len(accs) > 0:
        key = '-'.join([full_arg[o] for o in opt[:-1]])
        val = full_arg[opt[-1]]
        agg[ii][key].append((float(val), accs[-1]))
  handles_list = [h[1][0] for h in sorted(handles_dict.items())]
  labels_list = [h[1][1] for h in sorted(handles_dict.items())]
  print handles_list
  plt.legend(handles_list, labels_list, loc=4,prop={'size':17})
  #plt.title(desc)
  plt.xlabel('iteration', fontsize=24)
  plt.ylabel('accuracy', fontsize=24)
  plt.xlim([0,50])
  if 'trace2' in opts:
    plt.ylim([0,1])
  plt.tick_params(axis='x', labelsize=20)
  plt.tick_params(axis='y', labelsize=20)
  plt.tight_layout()
  plt.savefig('%s.pdf' % desc)

  # handle agg_opts
  for ii, aa in enumerate(agg):
    plt.clf()
    plt.hold(True)
    for k, v in aa.items():
      xys = sorted(v)
      xs = [xy[0] for xy in xys]
      ys = [xy[1] for xy in xys]
      plt.plot(xs, ys, label=k)
    plt.legend(loc=4,prop={'size':10})
    plt.savefig('%s_%s_%s.pdf' % (desc, '-'.join(agg_opts[ii][:-1]), agg_opts[ii][-1]))
