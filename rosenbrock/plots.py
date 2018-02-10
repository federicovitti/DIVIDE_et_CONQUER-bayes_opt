for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import sys
sys.path.append('/Users/vitti/Desktop/pierovitti/federico/STATISTICA/TESI/DetCBO/modules/BO')
sys.path.append('/Users/vitti/Desktop/pierovitti/federico/STATISTICA/TESI/DetCBO/modules/EBO')
sys.path.append('/Users/vitti/Desktop/pierovitti/federico/STATISTICA/TESI/DetCBO/modules/DetC')

from rosenbrock_bo import y_bo, err_bo, times_bo
from rosenbrock_ebo import y_EBO, err_EBO, times_EBO
from rosenbrock_my import y_bests, errs, times

import matplotlib.pyplot as plt
import numpy as np
import math



ys = np.vstack((y_bests, y_EBO, y_bo))
ers = np.vstack((errs, err_EBO, err_bo))
tis = np.vstack((times, times_EBO, times_bo))
cums = np.cumsum(ers, axis=2)

iters = 20
methods = list(['mondrian'])
methods.append('kmeans')
methods.append('dbscan')
methods.append('hdbscan')
methods.append('EBO')
methods.append('BO')


fig, (fig1, fig2) = plt.subplots(nrows=2, ncols=1, sharex=True)
fig.tight_layout(pad = 2, h_pad=3)
fig1.set_title('Average Simple Regret')

fig2.set_xlabel('iteration')
fig2.set_title('Average Cumulative Regret')
plt.xticks(range(20))


fig3, (fig4) = plt.subplots(1,1)
fig4.set_xlabel('time (s)')
fig4.set_title('Average Simple Regret')


for m in range(len(methods)):
	fig1.step(np.arange(iters), np.median(ers[m], axis=0), label=methods[m], linewidth=1)
	fig1.legend(loc=1, frameon=False)
	fig4.step(np.median(tis[m], axis=0), np.median(ers[m], axis=0), label=methods[m], linewidth=1)
	fig4.legend(loc=1, frameon=False) 
	fig2.step(np.arange(iters), np.median(cums[m], axis=0), label=methods[m], linewidth=1)

fig4_y = 0*np.ones(math.floor(np.mean(tis[:,:,-1], axis=1).max()))
fig1.plot(np.arange(iters), 0*np.ones(iters), color='gray', linestyle='--', linewidth=0.7)
fig4.plot(fig4_y, color='gray', linestyle='--', linewidth=0.7)




from openpyxl import Workbook
from openpyxl.compat import range
from openpyxl.utils import get_column_letter
wb = Workbook()
dest_filename = 'rosenbrock.xlsx'
ws1 = wb.active
ws1.title = "range names"
avg_ers = np.median(ers[:,:,-1], axis=1)
avg_tis = np.median(tis[:,:,-1], axis=1)
std_ers = 1.96*np.std(ers[:,:,-1], axis=1)
std_tis = 1.96*np.std(tis[:,:,-1], axis=1)

ws1.append(np.ndarray.tolist(avg_ers))
ws1.append(np.ndarray.tolist(avg_tis))
ws1.append(np.ndarray.tolist(std_ers))
ws1.append(np.ndarray.tolist(std_tis))

wb.save(filename = dest_filename)

fig.savefig('rosenERR.png')
fig3.savefig('rosenTIME.png')