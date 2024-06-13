#BEST       wall 2023-08-17 09:00:13.885511 n     10 k      1 LIFE       0.0000 ath    1100.0000 pop      3 m      1 r   1.00000000 node      2 back      1 leaves    1 depth    1 init    1 lmean 1100.00000000 pmean   5.00000000
# generate nbatch vs. accuracy matplotlib chart
import argparse
import numpy as np ; print('numpy ' + np.__version__)
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log',help='log file name',default=None)
parser.add_argument('--verbose', default=False, action='store_true')
args = parser.parse_args()
print(args)

def parselog(fn,field):
    f = open(fn,'r')
    a=[]
    while True:
        l = f.readline()
        if not l:
            break
        #if l[11:15] == 'wall':
        #if l[0:4] == 'BEST' or l[0:4] == 'LMAX':
        if l[0:4] == 'BEST' or l[0:4] == 'LMAX' or l[0:4] == 'BACK' or l[0:4] == 'ATH ':
            r = l[15:].split()
            #print(r)
            #print(r[field],r)
            try:
                a.append(float(r[field]))
            except:
                break

#            alphanumeric=''
#            for character in r[field]:
#                #if character.isdecimal() or character=='.':
#                if character in ['0','1','2','3','4','5','6','7','8','9','.']:
#                    alphanumeric += character
#            #print(alphanumeric,len(alphanumeric))
#            if len(alphanumeric)==0:
#                break
#            a.append(float(alphanumeric))
    return np.array(a[100:])

#BEST       wall 2022-10-12 11:31:04.221735 n      1 k      1 LIFE       0.0000 lmax       0.0000 pop      1 m      1 r   2.71800000 pmax      0
#BEST       wall 2022-12-18 10:42:55.454522 n      1 k      1 LIFE       0.0000 ath       0.0000 pop      1 m      1 r   1.00000000 nath      1 kmax      0 kmean   0.00000000 kstd   0.00000000 tol   1.00000000 lmax       0.0000
n = parselog(args.log,3)
k = parselog(args.log,5)
life = parselog(args.log,7)
lmax = parselog(args.log,9)
pop = parselog(args.log,11)
m = parselog(args.log,13)
r = parselog(args.log,15)
#nath = parselog(args.log,17)
node = parselog(args.log,17)
back = parselog(args.log,19)
leaf = parselog(args.log,21)
depth = parselog(args.log,23)
#prune = parselog(args.log,27)
lmean = parselog(args.log,25)
#lmax = parselog(args.log,29)
lstd = parselog(args.log,27)
ltop = parselog(args.log,29)
sib = parselog(args.log,31)
nsamp = parselog(args.log,33)
#uniq = parselog(args.log,33)
#pool = parselog(args.log,33)
# BEST       wall 2022-12-28 20:01:12.941516 n  24662 k      6 LIFE   20300.0000 ath   20300.0000 pop   1281 m      1 r   2.29175947 nath   5567 kmax     40 kmean   9.45000000 kstd   7.72447409   1.72552591 tol   1.00000000 lmax   20300.0000 khist    100
#d = parselog(args.log,21)

print(n.shape,n[0:10])

fig = plt.figure(figsize=(10,40))

#ax1 = fig.add_subplot(4,1,1)
#ax2 = fig.add_subplot(4,1,2,sharex=ax1)
#ax3 = fig.add_subplot(4,1,3,sharex=ax1)
#ax4 = fig.add_subplot(4,1,4,sharex=ax1)

#ax1 = fig.add_subplot(1,1,1)
ax1 = fig.add_subplot(3,1,1)
#ax11 = ax1.twinx()
#ax11.set_yscale('log')
ax3 = fig.add_subplot(3,1,2,sharex=ax1)

#ax1 = fig.add_subplot(3,1,1)
ax11 = ax1.twinx()
##ax11.set_yscale('log')
#ax2 = fig.add_subplot(3,1,2,sharex=ax1)
ax4 = fig.add_subplot(3,1,3,sharex=ax1)
ax41 = ax4.twinx()

#ax2 = fig.add_subplot(4,1,2, sharex = ax1)
#ax3 = fig.add_subplot(4,1,3, sharex = ax1)
#fig, (ax1,ax2,ax3) = plt.subplots(3,sharex=True,figsize=(10,40))
#fig, (ax1,ax2) = plt.subplots(2,sharex=True,figsize=(10,40))

#ax3 = ax2.twinx()
#ax1.plot(pop[0:len(n)],life[0:len(n)],'g-')
#ax1.set_ylim(auto=True)
#ax2.set_ylim(auto=True)
#ax3.set_ylim(auto=True)
#ax4.set_ylim(auto=True)

alpha = 0.5
ax1.scatter(n,pop[0:len(n)],marker='.',color='g',s=0.1,alpha=alpha)
#ax11.plot(n,sib[0:len(n)],'-k',linewidth=1)
#ax11.plot(n,r[0:len(n)],'-r',linewidth=1)
#ax1.scatter(n,pop[0:len(n)],marker='.',color='g',s=0.1,alpha=alpha)

#ax11.plot(n,uniq[0:len(n)],'-g',linewidth=1)

#ax1.plot(n,back[0:len(n)],'-m')
#ax1.plot(n,r[0:len(n)],'-g')
# ax2.plot(n,r[0:len(n)],'-k',linewidth=0.5)
#ax2.plot(n,pop[0:len(n)],'-r',linewidth=0.5,alpha=0.5)
#ax2.plot(n,r[0:len(n)],'-k')

#ax3.scatter(n,life[0:len(n)],marker='.',c=nsamp[0:len(n)],s=0.1)
ax3.scatter(n,life[0:len(n)],marker='.',color='b',s=0.1,alpha=alpha)
#ax3.plot(n, lmean[0:len(n)]+lstd[0:len(n)], '-k', linewidth=1)
ax3.plot(n, lmax[0:len(n)], '-k', linewidth=1)
#ax3.plot(n, ltop[0:len(n)], '-k', linewidth=1)
#ax3.plot(n, lbot[0:len(n)], '-k', linewidth=1)

ax4.scatter(n,sib[0:len(n)],marker='.',color='r',s=0.1,alpha=alpha)
#ax41.plot(n,leaf[0:len(n)],'-k',linewidth=1)
ax41.plot(n,node[0:len(n)],'-k',linewidth=1)

#ax3.plot(n, life[0:len(n)], '-b',linewidth=1,alpha=0.5)
#ax3.plot(n, lmean[0:len(n)], '-k', linewidth=1)
#ax3.plot(n, lmax[0:len(n)], '--k', linewidth=1)
#ax3.plot(n, lstd[0:len(n)], '-k', linewidth=1)
#ax4.scatter(n,k[0:len(n)],marker='.',color='k')
#ax4.plot(n, d[0:len(n)], 'k-')
#ax4.hist(k, bins=500, density=True, label='k')

#ax4.hist(k, bins=500, range=(0,500), density=True, label='k')
#ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: '{:.0f}'.format(x)))
#ax11.yaxis.set_major_formatter(FuncFormatter(lambda x, p: '{:6.0f}'.format(x)))
#ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: '{:5.0f}'.format(x)))

#ax3.plot(n,lmax,'r')

#ax2.set_xlabel('n')
ax3.set_xlabel('n')
ax1.set_xlabel('n')
ax1.set_ylabel('pop', color='g')
ax11.set_ylabel('depth', color='k')
ax41.set_ylabel('leaf', color='k')
#ax1.set_ylabel('uniq', color='g')
ax3.set_ylabel('life', color='b')
#ax2.set_ylabel('population', color='r')
ax4.set_ylabel('sample', color='r')
#ax1.text(0, position, label, transform=ax.transAxes, verticalalignment='center', color=color, fontsize=10)

#ax1.text(-0.075, 0.80, 'uniq', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,color='g',rotation=90)
#ax1.text(-0.075, 0.60, 'pool', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,color='c',rotation=90)
#ax1.text(-0.075, 0.40, 'leaf', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,color='k',rotation=90)
#ax1.text(-0.075, 0.20, 'depth', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,color='m',rotation=90)

#ax1.text(-0.1, 0.1, 'depth', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,color='m',rotation=90)


plt.show()


exit()

plt.xlabel('generation')
plt.plot(n,e,label='entropy')
plt.plot(n,pop,label='initial population')
plt.legend(loc='best')
plt.show()
exit()


#plt.figure(figsize=(40,10))
#ticks = np.arange(0,20000,40)
#plt.gca().set_yticks(ticks)
#plt.gca().set_xticks(ticks)
#plt.grid(True,color='red',linestyle='-')
xmax=10000000
ys=0.8
plt.xlim(0,xmax)
plt.ylim(ys,1.0)
plt.gca().set_yticks(np.arange(ys,1.0,0.1)) # major
plt.gca().set_yticks(np.arange(ys,1.0,0.01),minor=True)
plt.gca().set_xticks(np.arange(0,xmax,1000000))
plt.grid(True, which='major', axis='y', color='k', linestyle='-', linewidth=1)
plt.grid(True, which='minor', axis='y', color='k', linestyle='--', linewidth=0.5)

plt.plot(d0[:,0],d0[:,1],label='noise+raw_sc',linewidth=2)
plt.plot(d1[:,0],d1[:,1],label='noise+ref_sc',linewidth=2)
plt.plot(d2[:,0],d2[:,1],label='!noise+raw_sc',linewidth=2)
plt.plot(dref[:,0],dref[:,1],label='wifibaseband',linewidth=2)
#plt.plot(d30[:,0],d30[:,1],label='30M',linewidth=2)
#plt.plot(d20[:,0],d20[:,1],label='20M',linewidth=2)
#plt.plot(d10[:,0],d10[:,1],label='10M',linewidth=2)
#plt.axhline(y=0.97079729, color='red', linestyle=':',linewidth=5,label='wifiBaseband')
#plt.get_current_fig_manager().window.wm_geometry("+500+0")
plt.legend(loc='best')
plt.show()
