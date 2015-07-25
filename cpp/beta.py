import matplotlib.pyplot as plt
betas = []
for name in ['new3.txt']: #['big.txt']:#['out6.txt', 'tied.txt', 'beta_0.5.txt']:
  scores = []
  f = open(name, 'r')
  last_was_beta = False
  for line in f:
    toks = line.rstrip("\n").split()
    if last_was_beta:
      betas.append(map(float, toks))
      last_was_beta = False
    elif len(toks) > 0 and toks[0] == 'BETA:':
      last_was_beta = True
plt.matshow(betas[:30], vmax=5.0)
plt.show()
