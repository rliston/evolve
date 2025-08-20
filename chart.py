#BATCH            1 wall 2024-08-05 15:47:48.848148 lr 0.0001000000 wd 0.0100000000 batch     32 loss     1.675336     1.675336 grad     9.148475     9.148475
# generate nbatch vs. accuracy matplotlib chart
import argparse
import numpy as np ; print('numpy ' + np.__version__)
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--sigma', help='ylim',default=6.0, type=float)
#parser.add_argument('--start', help='start batch',default=0, type=int)
parser.add_argument('--log',help='log file name',default=None)
parser.add_argument('--verbose', default=False, action='store_true')
args = parser.parse_args()
print(args)

def parselog(fn):
    f = open(fn,'r')
    a=[]
    while True:
        l = f.readline()
        if not l:
            break
        if l[0:4] == 'BEST' or l[0:4] == 'LMAX' or l[0:4] == 'BACK' or l[0:4] == 'ATH ':
            b=[]
            for r in l[5:].split():
                try:
                    b.append(float(r))
                except:
                    b.append(0.0)
            a.append(b)
    return np.transpose(np.array(a))

print('loading log file')
arr = parselog(args.log)
print('arr',arr.shape)

#n=arr[4]
n=arr[28] # uniq
#n = np.arange(len(arr[4]))
k = arr[6]
life = arr[8]
lmax = arr[30]
lmin = arr[34]
pop = arr[12]
m = arr[14]
pmax = arr[16]
node = arr[18]
final = arr[20]
leaf = arr[22]
cmax = arr[24]
lmean = arr[26]
uniq = arr[28]
ltop = arr[30]
cmean = arr[32]
r = arr[34]
#pool = arr[36]
rad = arr[36]

print(n.shape,n[0:10])

plt.style.use('dark_background')

#fig = plt.figure(figsize=(10,40))
fig = plt.figure()

#ax1 = fig.add_subplot(4,1,1)
#ax2 = fig.add_subplot(4,1,2,sharex=ax1)
#ax3 = fig.add_subplot(4,1,3,sharex=ax1)
#ax4 = fig.add_subplot(4,1,4,sharex=ax1)

#ax1 = fig.add_subplot(1,1,1)
nplot=4
#ax1 = fig.add_subplot(nplot,1,1)
#ax11 = ax1.twinx()
#ax11.set_yscale('log')
#ax3 = fig.add_subplot(nplot,1,1,sharex=ax1)
ax3 = fig.add_subplot(nplot,1,1)

#ax1 = fig.add_subplot(3,1,1)
#ax11 = ax1.twinx()
##ax11.set_yscale('log')
#ax2 = fig.add_subplot(3,1,2,sharex=ax1)
ax4 = fig.add_subplot(nplot,1,3,sharex=ax3)
#ax44 = ax4.twinx()

ax5 = fig.add_subplot(nplot,1,2,sharex=ax3)
ax6 = fig.add_subplot(nplot,1,4,sharex=ax3)
#ax55 = ax5.twinx()


#ax41 = ax4.twinx()

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

alpha = 1.0
size = 1.0
ax6.scatter(n,final[0:len(n)],marker='.',color='y',s=size,alpha=alpha)
ax5.plot(n,cmean[0:len(n)],'-r',linewidth=1,alpha=alpha)
#ax5.scatter(n,rad[0:len(n)],marker='.',color='r',s=size,alpha=alpha)
#ax5.plot(n,rad[0:len(n)],'-k',linewidth=1,alpha=0.5)
#ax1.scatter(n,pop[0:len(n)],marker='.',color='g',s=0.1,alpha=alpha)
#ax1.scatter(n,pmax[0:len(n)],marker='.',color='g',s=0.1,alpha=alpha)
#ax1.plot(n,uniq[0:len(n)],'-k',linewidth=1,alpha=0.5)
#ax5.scatter(n,cmean[0:len(n)],marker='.',color='m',s=0.1,alpha=alpha)
# ax5.plot(n,cmean[0:len(n)],'-m',linewidth=1,alpha=0.5)
#ax5.scatter(n,cmax[0:len(n)],marker='.',color='k',s=0.1,alpha=0.2)
# ax55.plot(n,cmax[0:len(n)],'-k',linewidth=1,alpha=0.5)
#ax11.scatter(n,rad[0:len(n)],marker='.',color='k',s=0.1,alpha=alpha)
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
#ax3.plot(n, lmean[0:len(n)]+lstd[0:len(n)], '-k', linewidth=1)
#ax3.plot(n, lmax[0:len(n)], '-k', linewidth=1)
#ax3.scatter(n,lmax[0:len(n)],marker='.',color='k',s=0.1,alpha=alpha)
#ax3.scatter(n,ltop[0:len(n)],marker='.',color='k',s=0.1,alpha=alpha)
#ax3.scatter(n,lmean[0:len(n)],marker='.',color='k',s=0.1,alpha=alpha)
ax3.scatter(n,life[0:len(n)],marker='.',color='g',s=size,alpha=alpha)
ax3.plot(n, lmean[0:len(n)], '-g', linewidth=1)
ax3.plot(n, lmin[0:len(n)], '-g', linewidth=0.5)
ax3.plot(n, lmax[0:len(n)], '-g', linewidth=0.5)
#ax3.plot(n, ltop[0:len(n)], '-k', linewidth=1)
#ax3.plot(n, lbot[0:len(n)], '-k', linewidth=1)

#ax4.scatter(n,sib[0:len(n)],marker='.',color='r',s=0.1,alpha=alpha)
#ax41.plot(n,leaf[0:len(n)],'-k',linewidth=1)
#ax4.plot(n,node[0:len(n)],'-k',linewidth=1)
ax4.scatter(n,pop[0:len(n)],marker='.',color='m',s=size,alpha=alpha)
#ax4.plot(n,pop[0:len(n)],'-r',linewidth=1,alpha=0.5)
#ax4.scatter(n,pmax[0:len(n)],marker='.',color='g',s=0.1,alpha=alpha)
#ax4.scatter(n,pool[0:len(n)],marker='.',color='k',s=0.1,alpha=alpha)
#ax44.scatter(n,r[0:len(n)],marker='.',color='k',s=0.1,alpha=alpha)
#ax44.plot(n,uniq[0:len(n)],'-k',linewidth=1,alpha=0.5)
#ax44.scatter(n,m[0:len(n)],marker='.',color='k',s=0.1,alpha=alpha)
#ax4.scatter(n,uniq[0:len(n)],marker='.',color='k',s=0.1,alpha=alpha)

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
#ax1.set_xlabel('n')
#ax1.set_ylabel('uniq', color='g')
#ax11.set_ylabel('depth', color='k')
ax4.set_ylabel('pop', color='m')
#ax44.set_ylabel('uniq', color='k')
#ax1.set_ylabel('uniq', color='g')
ax3.set_ylabel('life', color='g')
ax5.set_ylabel('cmean', color='r')
ax6.set_ylabel('final', color='y')
# ax55.set_ylabel('cmax', color='k')
#ax2.set_ylabel('population', color='r')
#ax4.set_ylabel('sample', color='r')
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
exit()
lr=arr[5]
bloss=arr[11]
loss=arr[12]
bgrad=arr[14]
grad=arr[15]
#mom=arr[19]
#n=arr[0]
#lr=arr[5]
#bloss=arr[11]
#loss=arr[12]
#bgrad=arr[14]
#grad=arr[15]
#mom=arr[19]
#print('loss',np.mean(loss),np.std(loss))
#print('bloss',np.mean(bloss),np.std(bloss))
#print('grad',np.mean(grad),np.std(grad))
#print('bgrad',np.mean(bgrad),np.std(bgrad))

fig = plt.figure(figsize=(10,40))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2,sharex=ax1)
ax3 = fig.add_subplot(3,1,3,sharex=ax1)
#ax3b = ax3.twinx()
#ax1.set_ylim(np.mean(bloss)-args.sigma*np.std(bloss),np.mean(bloss)+args.sigma*np.std(bloss))
ax1.set_ylim(0,np.mean(bloss)+args.sigma*np.std(bloss))
ax1.scatter(n,bloss[0:len(n)],marker='.',color='k',s=0.1,alpha=0.1)
ax1.plot(n, loss[0:len(n)], '-g', linewidth=1)
ax1.axhline(y=np.min(bloss[0:len(n)]), color='green', linestyle=':',linewidth=1,label='min')
#ax1.scatter(n,loss[0:len(n)],marker='.',color='g',s=0.1,alpha=0.8)
#ax2.set_ylim(0,grad[args.start])
ax2.set_ylim(0,np.mean(bgrad)+args.sigma*np.std(bgrad))
ax2.scatter(n,bgrad[0:len(n)],marker='.',color='k',s=0.1,alpha=0.1)
ax2.plot(n, grad[0:len(n)], '-b', linewidth=1)
ax3.scatter(n,lr[0:len(n)],marker='.',color='r',s=0.2,alpha=1)
#ax3.plot(n,lr[0:len(n)], '-r', linewidth=1)
#ax3b.plot(n,mom[0:len(n)], '-m', linewidth=1)
#ax2.scatter(n,grad[0:len(n)],marker='.',color='b',s=0.1,alpha=0.8)
ax1.set_ylabel('loss', color='g')
ax2.set_ylabel('grad', color='b')
ax3.set_ylabel('lr', color='r')
ax3.set_xlabel('batch')
plt.show()

exit()
print('b',b.shape, b[0:10])
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
ax41.plot(n,pop[0:len(n)],'-k',linewidth=1)

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
plt.grid(True, which='major', axis='y', color='grey', linestyle='-', linewidth=1)
plt.grid(True, which='minor', axis='y', color='grey', linestyle='--', linewidth=0.5)

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
