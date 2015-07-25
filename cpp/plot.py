import matplotlib.pyplot as plt
all_scores = []
for name in ['big.txt', 'big_relaxed.txt']:#['out6.txt', 'tied.txt', 'beta_0.5.txt']:
  scores = []
  f = open(name, 'r')
  for line in f:
    toks = line.rstrip("\n").split()
    if len(toks) > 0 and toks[0] == 'Trace:':
      scores.append(float(toks[1]))
  all_scores.append(scores)
plt.hold(True)
plt.plot(all_scores[0], 'b', label='All')
plt.plot(all_scores[1], 'r', label='None')
#plt.plot(all_scores[1], 'r', label='One')
#plt.plot(all_scores[2], 'g', label='Fixed')
plt.legend()
plt.show()
