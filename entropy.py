import argparse
import random
import datetime
import time
import numpy as np ; print('numpy ' + np.__version__)
import lifelib ; print('lifelib',lifelib.__version__)
import scipy.stats
import os
import copy

np.set_printoptions(linewidth=250)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--init', help='initial pattern', default=None, type=str)
#parser.add_argument('--density', help='density target', default=0.0, type=float)
#parser.add_argument('--pos', help='positive density target', default=0.0, type=float)
#parser.add_argument('--neg', help='negative density target', default=0.0, type=float)
parser.add_argument('--rad', help='radius', default=50, type=float)
parser.add_argument('--bang', help='initial population factor', default=0.0, type=float)
parser.add_argument('--flip_pos', help='k threshold to flip from 0 to 1 density', default=5, type=int)
parser.add_argument('--flip_neg', help='k threshold to flip from 1 to 0 density', default=5, type=int)
parser.add_argument('--direction', help='starting direction 1,-1', default=1, type=int)
#parser.add_argument('--advance', help='# gen to advance on ATH', default=1, type=int)
#parser.add_argument('--mode_pop', default=False, action='store_true')
#parser.add_argument('--mode_mut', default=False, action='store_true')
#parser.add_argument('--mode_ath', default=False, action='store_true')
#parser.add_argument('--maxpop', help='param for mode_pop', default=1000, type=int)
#parser.add_argument('--nmut', help='param for mode_mut', default=1000, type=int)
#parser.add_argument('--patience', help='# mut without ath before ash', default=10000, type=int)
#parser.add_argument('--effort', help='number of mutations to try before growing radius', default=10000, type=int)
#parser.add_argument('--prefill', help='prefill radius', default=0, type=float)
#parser.add_argument('--decay', help='lmax decay per n', default=1.000000, type=float)
#parser.add_argument('--tol', help='tolerance', default=0.99, type=float)
#parser.add_argument('--grate', help='growth rate', default=100, type=float)
#parser.add_argument('--step', help='# gen per step for entropy calculation', default=1, type=int)
#parser.add_argument('--timeout', help='timeout in units of n', default=10000, type=int)
#parser.add_argument('--gen', help='initial generation', default=0, type=int)
#parser.add_argument('--period', help='length of population entropy trace', default=1000, type=int)
#parser.add_argument('--gamma', help='radius coefficient', default=1.414, type=float)
parser.add_argument('--seed', help='random seed', default=None, type=int)
parser.add_argument('--memory', help='garbage collection limit in MB', default=60000, type=int)
parser.add_argument('--results', help='results directory', default='./results')
parser.add_argument('--choice',help='select from pat.coords[]',default=False, action='store_true')
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

#def log(hdr,n,e,lmax,m,pop,r,l,pmax,k):
def log(hdr,n,k,l,ath,pop,m,r,pmax,nath,d):
#    bb = pat.bounding_box
#    if bb is not None:
#        d = pat.population / (np.pi*(min(bb[2:])*0.5)*(min(bb[2:])*0.5))
#        if d>1:
#            d=0
#    else:
#        d=0
    print('{:10} wall {} n {:6d} k {:6d} LIFE {:12.4f} ath {:12.4f} pop {:6d} m {:6d} r {:12.8f} pmax {:6d} nath {:6d} density {:12.8f} {: 3d}'.format(hdr,datetime.datetime.now(),n,k,l,ath,pop,m,r,pmax,nath,d,direction))
    #print('{:10} wall {} n {:6d} k {:6d} LIFE {:12.4f} ath {:12.4f} pop {:6d} m {:6d} r {:12.8f} pmax {:6d} nath {:6d} ath_ash {:6d} patience {:6d}'.format(hdr,datetime.datetime.now(),n,k,l,ath,pop,m,r,pmax,nath,ath_ash.population,args.patience))

def radius(pop, n):
    return args.rad
    #return args.density*np.sqrt(pop/np.pi) # pi*r^2 = pop

    #return args.rad+(np.sqrt(n)/args.grate)
    #return 1+(n/10000000.)
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
        #o = int(max(0,lmax-10*period))
        o = int(max(0,lmax-period))
        pat = pat.advance(o) # jump to lmax
        p0 = pat.population
        for j in range(1000000//period):
            for k in range(period):
                pt[k] = pat.population
                pat = pat.advance(1)
            if (pat.population>(2*p0)) and ((pat.population-p0)>10000): # 2x pop growth over lmax pop is suspicious
                return -2 # linear growth
            #if n > 12700:
            #    print('n',n,p0,pat.population-p0)
            #if (pat.population/p0) > 10.0:
            #    return -1 # 10x pop growth over lmax is suspicious
            #pt = [pat.advance(k).population for k in range(1,period)] # population trace up to period
            #pt = [pat[o+k].population for k in range(period)] # population trace up to period
            value,counts = np.unique(pt, return_counts=True) # histogram of population trace
            e = scipy.stats.entropy(counts,base=None) # entropy of population distribution 
            #print(e,pt)
            if e<0.682689492*np.log(period): # threshold
                return max(0,o+j*period)
        return -1 # runaway


def ash(pat,lmax):
    pat = pat.advance(lmax)
    pat = pat.advance(100000) # allow gliders to escape
    pat = pat[-10000:10000, -10000:10000]
    pat = pat.centre()
    return pat

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
    pat = pat.centre()

lmax=0
pmax=0
ath=0
n=0
k=0
nath=0
direction=args.direction
#ath_fn=None
#ath_ash = copy.copy(pat)
#t0=time.time()
#n0=0
#h=[]
#fn=None

#while pat.population < (args.prefill*args.prefill*np.pi)/0.682689492:
#    pat[np.random.normal(0,args.prefill,size=1)] |= 1;
#pat[np.random.normal(0,args.rad,size=[int(args.rad*args.rad*np.pi*args.density),2])] |= 1
pat[np.random.normal(0,args.rad,size=[int(args.rad*args.rad*np.pi*args.bang),2])] |= 1
print('bang population',pat.population)
r = args.rad
while True:
    n+=1
    k+=1
    nath+=1

#    if nath>args.patience:
#        log('PATIENCE',n,k,l,ath,pat.population,len(xy),r,pmax,nath)
#        fn = '{}/patience_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,pmax,int(lmax),args.seed,n)
#        bb = pat.bounding_box
#        if bb is not None:
#            pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
#        #pat = ash(pat,lmax)
#        pat = copy.copy(ath_ash)
#        #if ath_fn is not None:
#        #    pat = lt.load(ath_fn) # empty pattern if None, else load .rle file
#        #    pat = pat.centre()
#        lmax=0
#        l = lifespan(pat,100,lmax)
#        nath=0

#        bb = pat.bounding_box
#        if bb is not None:
#            args.rad = 0.5*min(bb[2:])*args.density
            #args.rad = 0.5*max(bb)*args.density
            #args.rad = 0.5*np.sqrt(bb[2]*bb[2]+bb[3]*bb[3])*args.density

#    if args.mode_mut and (n%args.nmut)==0:
#        #adv = args.advance
#        #adv = n//args.nmut
#        #pat = pat.advance(adv)
#        #lmax -= adv
#        log('MUT',n,k,l,ath,pat.population,len(xy),r,pmax,nath)
#        fn = '{}/mut_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,pmax,int(lmax),args.seed,n)
#        bb = pat.bounding_box
#        pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
#        #if lmax==ath:
#        #    athpat = lt.pattern(ash(pat,lmax).coords())
#        #    print('athpat pop',athpat.population)
#        #pat=athpat
#        #print('pat pop',pat.population)
#        pat = ash(pat,lmax)
#        #pat = lt.load(athfn) # empty pattern if None, else load .rle file
#        #pat = pat.centre()
#        lmax=0

    #r0 = radius(pmax,n)
    r0 = radius(pat.population,n)
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

    # estimate current density
    bb = pat.bounding_box
    if bb is not None:
        d = pat.population / (np.pi*(min(bb[2:])*0.5)*(min(bb[2:])*0.5))
        if d>1:
            d=0
    else:
        d=0

    xy=[]
    v=[]
#    if args.choice:
#        #xy = np.random.normal(0,r,size=[int(np.ceil(random.expovariate(1))),2])
#        coords = pat.coords()
#        if len(coords)>0:
#            xy.extend([random.choice(coords)])
#            v.extend([0])
#            #xy.extend(np.random.choice(coords))
#            #v.append(0)
    #else:
    #    xy = np.random.normal(0,r,size=[int(np.ceil(random.expovariate(1))),2])
    #xy = coords[np.random.choice(len(coords), size=[1])]
    #print('xy.shape',xy.shape,xy)
    #v0 = pat[xy]
    #print('v0.shape',v0.shape,v0)
    #pat[xy]=[0]*len(xy)

    # randomly make cells alive
    ##xy = [np.random.normal(0,r,size=2) for k in range(m)]
    ##v = [(np.random.uniform() < args.pos) for k in range(m)]
    #m = int(np.ceil(random.expovariate(1)))

    if direction<0 and k>args.flip_pos:
        direction = 1
        log('POS',n,k,l,ath,pat.population,len(xy),r,pmax,nath,d)
        fn = '{}/pos_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,pat.population,int(lmax),args.seed,n)
        bb = pat.bounding_box
        pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
        k=0
    elif direction>0 and k>args.flip_neg:
        direction = -1
        log('NEG',n,k,l,ath,pat.population,len(xy),r,pmax,nath,d)
        fn = '{}/neg_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,pat.population,int(lmax),args.seed,n)
        bb = pat.bounding_box
        pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
        k=0

    if direction == -1:
        coords = pat.coords()
        if len(coords)>0:
            xy.extend([random.choice(coords)])
            v.extend([0])
    else:
        m = int(np.ceil(random.expovariate(1)))
        m0=m
        while m0>0:
            xy0 = np.random.normal(0,r,size=[1,2])
            if pat[xy0]==1:
                continue
            else:
                xy.extend(xy0)
                v.extend([1]) # try to approach args.density
                m0-=1
                
        #xy.extend(np.random.normal(0,r,size=[m,2]))
        #v.extend([1]*m) # try to approach args.density


#    if nath > args.flip:
#        nath=0
#        if args.density==1:
#            args.density=0
#        else:
#            args.density=1
#
#    if d > args.density:
#        if args.choice:
#            coords = pat.coords()
#            if len(coords)>0:
#                xy.extend([random.choice(coords)])
#                v.extend([0])
#        else:
#            m = int(np.ceil(random.expovariate(1)))
#            xy.extend(np.random.normal(0,r,size=[m,2]))
#            v.extend([0]*m) # try to approach args.density
#    else:
#        m = int(np.ceil(random.expovariate(1)))
#        xy.extend(np.random.normal(0,r,size=[m,2]))
#        v.extend([1]*m) # try to approach args.density

#    if k>10:
#        m = int(np.ceil(random.expovariate(1)))
#        xy.extend(np.random.normal(0,r,size=[m,2]))
#        v.extend([1]*m)

#    for i in range(int(np.ceil(random.expovariate(1)))):
#        xy.extend(np.random.normal(0,r,size=[1,2]))
#        v.extend([int(d < args.density)]) # try to approach args.density
#        #v.extend([int(np.random.uniform() < args.density)])
        
#    if np.random.uniform() < args.pos:
#        xy.extend(np.random.normal(0,r,size=[1,2]))
#        v.extend([1])

    # randomly make cells dead
#    if np.random.uniform() < args.neg:
#        xy.extend(np.random.normal(0,r,size=[1,2]))
#        v.extend([0])
    #m = int(np.ceil(random.expovariate(1)))
    #xy.extend(np.random.normal(0,r,size=[m,2]))
    #v.extend(~(np.random.uniform(size=m) < args.neg))

    xy = np.array(xy)
    v = np.array(v)
    #print('xy',xy,'v',v)
    v0 = pat[xy]
    pat[xy] = v # apply mutation xy
#    if (sum(v)>0):
#        print('v',v,'v0',v0)
    #pat[xy] ^=1 # apply mutation xy
    l = lifespan(pat,100,lmax)
    if l==-1:
        log('RUNAWAY',n,k,l,ath,pat.population,len(xy),r,pmax,nath,d)
        bb = pat.bounding_box
        pat.write_rle('{}/runaway_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
        pat[xy] = v0 # revert mutation xy
        #pat[xy] ^=1 # revert mutation xy
        bb = pat.bounding_box
        pat.write_rle('{}/snapshot_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
        #lmax *= asteroid
    elif l==-2:
        log('GROWTH',n,k,l,ath,pat.population,len(xy),r,pmax,nath,d)
        bb = pat.bounding_box
        pat.write_rle('{}/growth_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
        #pat[xy] ^=1 # revert mutation xy
        pat[xy] = v0 # revert mutation xy
        bb = pat.bounding_box
        pat.write_rle('{}/snapshot_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
    #elif (l>=(lmax-800)) and (l<=lmax):
    elif l==lmax:
        log('BEST',n,k,l,ath,pat.population,len(xy),r,pmax,nath,d)
        k=0
    elif l>lmax:
        lmax=l
        nath=0
        if lmax>ath:
            ath = lmax
#            if args.mode_ath:
#                pat = ash(pat,lmax)
#                lmax=0
            log('ATH',n,k,l,ath,pat.population,len(xy),r,pmax,nath,d)
            fn = '{}/ath_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,pat.population,int(lmax),args.seed,n)
            bb = pat.bounding_box
            pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)

#            ath_ash = copy.copy(ash(pat,lmax))
#            args.patience = ath_ash.population
#            bb = ath_ash.bounding_box
#            if bb is not None:
#                args.rad = 0.5*min(bb[2:])*args.density
            #ath_fn = '{}/ash_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,pmax,int(lmax),args.seed,n)
            #bb = ath_ash.bounding_box
            #ath_ash.write_rle(ath_fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)

            #pat = pat.advance(args.advance)
            #lmax -= args.advance
##            pt = np.zeros(args.advance)
##            pa = []
##            for k in range(args.advance):
##                pt[k] = pat.population
##                pa.append(pat)
##                pat = pat.advance(1)
##            k = np.argmin(pt)
##            pat = pa[k]
#            glider = 4*r
#            coords = pat.coords()
#            pat = lt.pattern()
#            for j in range(len(coords)):
#                if (abs(coords[j,0]) < glider) and (abs(coords[j,1]) < glider):
#                    pat[coords[j,0],coords[j,1]] = 1
##            lmax -= k
        else:
            log('LMAX',n,k,l,ath,pat.population,len(xy),r,pmax,nath,d)
            #fn = '{}/lmax_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,pmax,int(lmax),args.seed,n)
            #bb = pat.bounding_box
            #pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
        #fn = '{}/ath_P{:06d}_L{:06d}_seed{:09d}_n{:09d}_init.rle'.format(args.results,pmax,int(lmax),args.seed,n)
        #pat.write_rle(fn, header=None, footer=None, comments=str(args), file_format='rle', save_comments=True)
        k=0
    else:
        pat[xy] = v0 # revert mutation xy
        #pat[xy] ^=1 # revert mutation xy
        #lmax *= asteroid

    if pat.population > pmax:
        pmax = pat.population
#        if args.mode_pop:
#            pt = np.zeros(args.advance)
#            pa = []
#            for k in range(args.advance):
#                pt[k] = pat.population
#                pa.append(pat)
#                pat = pat.advance(1)
#            k = np.argmin(pt)
#            pat = pa[k]
#            lmax -= k
#            log('POP',n,k,l,ath,pat.population,len(xy),r,pmax)
#            if pat.population>0:
#                fn = '{}/pop_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,pmax,int(lmax),args.seed,n)
#                bb = pat.bounding_box
#                pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)

        #log('POP',n,lmax,lmax,len(xy),pat.population,r,ath,pmax,k)
        #fn = '{}/pop_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,pmax,int(lmax),args.seed,n)
        #pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)

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
