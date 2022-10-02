# generate nbatch vs. accuracy matplotlib chart
import argparse
import numpy as np ; print('numpy ' + np.__version__)
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

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
        if l[0:4] == 'BEST':
            r = l[15:].split()
            #print(r)
            #print(r[field],r)
            alphanumeric=''
            for character in r[field]:
                if character.isdecimal():
                    alphanumeric += character
            #print(alphanumeric,len(alphanumeric))
            if len(alphanumeric)==0:
                break
            a.append(float(alphanumeric))
    return np.array(a)

n = parselog(args.log,3)
k = parselog(args.log,5)
ent = parselog(args.log,9)
life = parselog(args.log,7)
pop = parselog(args.log,11)
#bt = parselog(args.log,9+8)
r = parselog(args.log,15)
r0 = parselog(args.log,19)

print(n.shape,n[0:10])

fig = plt.figure(figsize=(10,40))
ax1 = fig.add_subplot(4,1,1)
ax2 = fig.add_subplot(4,1,2, sharex = ax1)
ax3 = fig.add_subplot(4,1,3, sharex = ax1, sharey = None)
ax4 = fig.add_subplot(4,1,4)

#fig, (ax1,ax2,ax3) = plt.subplots(3,sharex=True,figsize=(10,40))
#fig, (ax1,ax2) = plt.subplots(2,sharex=True,figsize=(10,40))

#ax3 = ax2.twinx()
#ax1.plot(pop[0:len(n)],life[0:len(n)],'g-')
ax1.plot(n,r0[0:len(n)],'g-')
ax2.plot(n,pop[0:len(n)],'r-')
ax3.plot(n, life[0:len(n)], 'b-')
ax4.hist(k, bins=500, range=(0,5000), density=True, label='k')
#ax3.plot(n,lmax,'r')

ax2.set_xlabel('n')
ax3.set_xlabel('n')
ax1.set_xlabel('n')
ax1.set_ylabel('radius', color='g')
ax3.set_ylabel('lifespan', color='b')
ax2.set_ylabel('population', color='r')
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
