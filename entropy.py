import argparse
import random
import datetime
import time
import numpy as np ; print('numpy ' + np.__version__)
import lifelib ; print('lifelib',lifelib.__version__)
import scipy.stats
import os

np.set_printoptions(linewidth=250)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--init', help='initial pattern', default=None, type=str)
parser.add_argument('--density', help='density target', default=0.682689492, type=float)
#parser.add_argument('--effort', help='number of mutations to try before growing radius', default=10000, type=int)
#parser.add_argument('--prefill', help='prefill radius', default=0, type=float)
#parser.add_argument('--decay', help='lmax decay per n', default=1.000000, type=float)
#parser.add_argument('--tol', help='tolerance', default=0.99, type=float)
parser.add_argument('--rad', help='minimum radius', default=1, type=float)
#parser.add_argument('--step', help='# gen per step for entropy calculation', default=1, type=int)
#parser.add_argument('--timeout', help='timeout in units of n', default=10000, type=int)
#parser.add_argument('--gen', help='initial generation', default=0, type=int)
#parser.add_argument('--period', help='length of population entropy trace', default=1000, type=int)
#parser.add_argument('--gamma', help='radius coefficient', default=1.414, type=float)
parser.add_argument('--seed', help='random seed', default=None, type=int)
parser.add_argument('--memory', help='garbage collection limit in MB', default=60000, type=int)
parser.add_argument('--results', help='results directory', default='./results')
#parser.add_argument('--ath', help='only save all time high patterns', default=False, action='store_true')
#parser.add_argument('--watchdog', help='timeout parameter', default=2.0, type=float)
#parser.add_argument('--minrad', help='minimum radius', default=1.0, type=float)
#parser.add_argument('--gamma', help='radius coefficient', default=3.14, type=float)
##parser.add_argument('--expo', help='exponential distribution parameter', default=0, type=float)
#parser.add_argument('--ipop', help='initial population', default=0, type=int)
##parser.add_argument('--srad', help='std dev for new grid entries', default=100, type=float)
##parser.add_argument('--spop', help='initial grid entries', default=10, type=int)
##parser.add_argument('--patience', help='number of batches before backtracking', default=100, type=int)
#parser.add_argument('--backtrack', help='number of batches before backtracking', default=0, type=int)
#parser.add_argument('--pattern', help='initial pattern', default=None, type=str)
#parser.add_argument('--seed', help='random seed', default=None, type=int)
#parser.add_argument('--memory', help='garbage collection limit in MB', default=60000, type=int)
#parser.add_argument('--results', help='results directory', default='./results')
##parser.add_argument('--space', help='grid spacing', default=25, type=int)
##parser.add_argument('--side', help='grid size = side*2+1', default=2, type=int)
##parser.add_argument('--radius', help='initial radius', default=1.0, type=float)
#parser.add_argument('--period', help='population trace length', default=100, type=int)
#parser.add_argument('--batch', help='number of mutations / batch', default=100, type=int)
##parser.add_argument('--terminate', default=False, action='store_true')
##parser.add_argument('--bump', default=False, action='store_true')
#
###parser.add_argument('--space', help='grid spacing', default=100, type=int)
##parser.add_argument('--timeout', help='stopping limit', default=99999999, type=int)
##parser.add_argument('--keep', help='harmless mutation rate', default=None, type=float)
###parser.add_argument('--advance', help='step size', default=100, type=int)
##parser.add_argument('--sigma', help='sample area for density calculation', default=3, type=float)
parser.add_argument('--summary',help='only save final pattern',default=False, action='store_true')
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
if args.seed is None:
    args.seed = random.randint(1,1000000)
random.seed(args.seed)
np.random.seed(args.seed)
#args.ipop = int(args.radius*args.radius)
#args.batch = int(args.radius)
print(args)

def log(hdr,n,e,lmax,m,pop,r,l,r0,k,rank=0):
    print('{:10} wall {} n {:6d} k {:6d} LIFE {:12.4f} lmax {:12.4f} pop {:6d} m {:6d} r {:12.8f} rank {:6d} r0 {:12.8f}'.format(hdr,datetime.datetime.now(),n,k,e,lmax,pop,m,r,rank,r0))

def radius(pop):
    return args.density*np.sqrt(pop/np.pi) # pi*r^2 = pop
    #return np.sqrt(((1-args.density)*pop)/np.pi) # pi*r^2 = 0.68*pop
    #return max(args.rad,np.sqrt(((1-args.density)*pop)/np.pi)) # pi*r^2 = 0.68*pop
    #return min(args.rad,np.sqrt((0.682689492*pop)/np.pi)) # pi*r^2 = 0.68*pop, CAP AT args.rad
    #return max(args.rad,np.sqrt((args.density*pop)/np.pi)) # pi*r^2 = 0.68*pop
    #return max(args.rad,np.sqrt(pop/np.pi))
    #return max(args.rad,np.sqrt(pop))
    #return args.rad
    #return max(args.rad,args.gamma*np.sqrt(pop))
    #return max(1.414,args.gamma*np.sqrt(pop))

# run soup until population is stable, starting at generation lmax
def lifespan(pat,period,lmax):
        pt = np.zeros(period)
        o = int(max(0,lmax-period))
        pat = pat.advance(o) # jump to lmax
        for j in range(1000000//period):
            for k in range(period):
                pt[k] = pat.population
                pat = pat.advance(1)
            #pt = [pat.advance(k).population for k in range(1,period)] # population trace up to period
            #pt = [pat[o+k].population for k in range(period)] # population trace up to period
            value,counts = np.unique(pt, return_counts=True) # histogram of population trace
            e = scipy.stats.entropy(counts,base=None) # entropy of population distribution 
            #print(e,pt)
            if e<0.682689492*np.log(period): # threshold
                return max(0,o+j*period)
        return -1 # runaway

## compute population entropy from gen:gen+period
#def entropy(pat,period,gen):
#    trace = np.zeros(period)
#    #pat = pat.advance(gen) # jump to lmax
#    for k in range(period//args.step):
#        #print('k',k,'pop',pat.population,'components',len(pat.components()))
#        trace[k] = pat.population
#        #trace[k] = len(pat.components())
#        pat = pat.advance(args.step)
#    #return np.std(trace)
#    #return np.mean(trace)
#    value,counts = np.unique(trace, return_counts=True) # histogram of population trace
#    return scipy.stats.entropy(counts) # entropy of population distribution 

sess = lifelib.load_rules("b3s23")
lt = sess.lifetree(memory=args.memory)
#pat = lt.pattern()
if args.init is None:
    pat = lt.pattern()
else:
    pat = lt.load(args.init) # empty pattern if None, else load .rle file

lmax=0
ath=0
n=0
k=0
#t0=time.time()
#n0=0
#h=[]
fn=None

#while pat.population < (args.prefill*args.prefill*np.pi)/0.682689492:
#    pat[np.random.normal(0,args.prefill,size=1)] |= 1;
#pat[np.random.normal(0,args.prefill,size=[int((args.prefill*args.prefill*np.pi)/0.682689492),2])] |= 1
#print('prefill pop',pat.population)
r = args.rad
while True:
    n+=1
    k+=1
    r0 = radius(pat.population)
    r = max(args.rad,r0)
#    if k>args.effort:
#        r = radius(pat.population)
#        log('GROW',n,l,lmax,len(xy),pat.population,r,ath,0,k)
#        k=0
#        fn = '{}/grow_E{:08.0f}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,lmax,int(lmax),args.seed,n)
#        pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
#    else:
#        k+=1
    #asteroid = np.random.standard_exponential()
    #lmax /= asteroid
    xy = np.random.normal(0,r,size=[int(np.ceil(random.expovariate(1))),2])
    pat[xy] ^=1 # apply mutation xy
    l = lifespan(pat,100,lmax)
    if l<0:
        log('RUNAWAY',n,l,lmax,len(xy),pat.population,r,ath,0,k)
        bb = pat.bounding_box
        pat.write_rle('{}/runaway_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
        pat[xy] ^=1 # revert mutation xy
        bb = pat.bounding_box
        pat.write_rle('{}/snapshot_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
        #lmax *= asteroid
    elif l>=lmax:
        log('BEST',n,l,lmax,len(xy),pat.population,r,ath,r0,k)
        k=0
        lmax=l
        bb = pat.bounding_box
        if lmax>ath:
            log('ATH',n,lmax,lmax,len(xy),pat.population,r,ath,r0,k)
            ath = lmax;
            fn = '{}/ath_E{:08.0f}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,lmax,int(lmax),args.seed,n)
            pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
            fn = '{}/ath_E{:08.0f}_L{:06d}_seed{:09d}_n{:09d}_init.rle'.format(args.results,lmax,int(lmax),args.seed,n)
            pat.write_rle(fn, header=None, footer=None, comments=str(args), file_format='rle', save_comments=True)
    else:
        pat[xy] ^=1 # revert mutation xy
        #lmax *= asteroid

#    b=[]
#    for k in range(args.batch):
#        n+=1
#        xy = np.random.normal(0,r,size=[int(np.ceil(random.expovariate(1))),2])
#        pat[xy] ^=1 # apply mutation xy
#        l = lifespan(pat,100,0)
#        if l<0:
#            log('RUNAWAY',n,l,lmax,len(xy),pat.population,r,ath,0,n)
#            bb = pat.bounding_box
#            pat.write_rle('{}/runaway_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
#        else:
#            b.append((l,xy)) # list of tuples
#        pat[xy] ^=1 # revert mutation xy
#
#    b.sort(reverse=True,key=lambda y: y[0])
#    rank = int(np.ceil(random.expovariate(1)))
#    (bl,bxy) = b[rank]
#    pat[bxy] ^=1 # apply best mutation xy
#    lmax=bl
#    log('BEST',n,lmax,lmax,len(bxy),pat.population,r,ath,0,n,rank)
#    bb = pat.bounding_box
#    if lmax>ath:
#        ath = lmax;
#        log('ATH',n,lmax,lmax,len(bxy),pat.population,r,ath,0,n,rank)
#        fn = '{}/ath_E{:08.0f}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,lmax,int(lmax),args.seed,n)
#        pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)



#    n0=n
        
#    #xy = np.random.normal(0,r,size=[int(np.ceil(random.expovariate(1/r))),2])
#    xy = np.random.normal(0,r,size=[int(np.ceil(random.expovariate(1))),2])
#    # apply mutation xy
#    pat[xy] ^=1
#    #e = entropy(pat,args.period,args.gen)
#    edist =  np.random.normal(loc=0, scale=args.tol)
#    etol = emax*(1.+(args.gamma*args.tol)+edist)
#    #etol = emax*args.tol
#    #etol = emax * (1. - (1./np.sqrt(1.+n)))
#    #etol = emax * (1. - (1./r))
#    e = lifespan(pat,100,int(etol))
#    #h.append(e)
#    #if e>emax*args.tol:
#    #etol = 2*np.mean(h)*args.tol
#    if e>=etol:
#        #l = lifespan(pat,100,0)
#        emax=e
#        #if e>emax:
#        #    emax=e
#        #args.gen = max(0,l-args.period)
#        #args.gen = l
#        #t0=time.time()
#        log('BEST',n,e,emax,len(xy),pat.population,r,e,etol,n-n0)
#        bb = pat.bounding_box
#        if fn is not None:
#            os.remove(fn)
#        fn = '{}/best_E{:08.0f}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,e,int(e),args.seed,n)
#        pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
#        if e>ath:
#            pat.write_rle('{}/ath_E{:08.0f}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,e,int(e),args.seed,n), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
#            log('ATH',n,e,emax,len(xy),pat.population,r,e,etol,n-n0)
#            ath=e
#        n0=n
#    else:
#        pat[xy] ^=1 # revert
#    n+=1
#    #if time.time() > t0+args.timeout:
#    if n > n0+args.timeout:
#        print('{:10} wall {} n {:6d}'.format('TIMEOUT',datetime.datetime.now(),n))
#        break
