import argparse
import random
import datetime
import numpy as np ; print('numpy ' + np.__version__)
import lifelib ; print('lifelib',lifelib.__version__)

np.set_printoptions(linewidth=250)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--prefill',help='sparse prefill radius',default=0,type=int)
parser.add_argument('--alpha', help='prefill density', default=1.0, type=float)
parser.add_argument('--keep', help='fraction of harmless mutations to retain [0,1]', default=1.0, type=float)
parser.add_argument('--results', help='results directory', default='./results')
parser.add_argument('--seed', help='random seed', default=None, type=int)
parser.add_argument('--memory', help='garbage collection limit in MB', default=30000, type=int)
parser.add_argument('--patience', help='generations to wait before reset', default=10000, type=int)
parser.add_argument('--summary',help='only save final pattern',default=False, action='store_true')
parser.add_argument('--verbose', default=False, action='store_true')
args = parser.parse_args()
if args.seed is None:
    args.seed = random.randint(1,1000000)
random.seed(args.seed)
print(args)

def log(hdr,n,k,l,m,pop,d,r,patience,keep):
    print('{:10} wall {} k {:6d} n {:6d} LIFE {:6d} pop {:6d} m {:6d} r {:12.8f} density {:12.8f} patience {:12.0f} keep {:12.8f}'.format(hdr,datetime.datetime.now(),k,n,l,pop,m,r,d,patience,keep))

# run soup until population is stable
def lifespan(pat):
        last=0
        itot=0
        for i in range(10000):
            pat = pat.advance(100+(i%7))
            itot += 100+(i%7)
            if pat.population == last:
                return itot
            else:
                last=pat.population
        print('RUNAWAY ...')
        pat.centre().save('{}/runaway_seed{}_d{}_gen{}_pop{}.rle'.format(args.results,args.seed,pat.population / (pat.bounding_box[2]*pat.bounding_box[3]),itot,pat.population))
        return 0

sess = lifelib.load_rules("b3s23")
lt = sess.lifetree(memory=args.memory) # 50GB RAM
pat = lt.pattern() # empty pattern

lmax=0
n=0
k=0

# prefill WIP
pat[0,0] = 1 # seed
r=args.prefill
xy = np.random.normal(0,r,size=(int(args.alpha*r*r),2))
xy = np.around(xy)
xy = xy.astype(int)
#print('xy.shape',xy.shape)
pat[xy] ^= 1

patience = args.patience
while True:
    n+=1
    k+=1
    #r = np.sqrt(pat.population) # radius
    #r = 0.6827*np.sqrt(pat.population) # radius
    r = args.prefill

    # apply random mutations
    m = random.expovariate(1)
    m = int(np.ceil(m))
    xy=[(int(random.normalvariate(0,r)),int(random.normalvariate(0,r))) for k in range(m)]
    for (x,y) in xy:
        pat[x,y] ^= 1

    # early mutations can result in emptiness
    if pat.bounding_box is None:
        continue
    d = pat.population / (pat.bounding_box[2]*pat.bounding_box[3]) # density
    #d = pat.population / (np.pi*(3*r)*(3*r)) # density at 3 sigma
    keep = args.keep

    # use lifelib to compute lifespan
    l = lifespan(pat)

    if l>lmax:
        log('BEST',n,k,l,m,pat.population,d,r,patience,keep)
        if not args.summary:
            pat.centre().save('{}/best_L{:09d}_P{:04d}_K{:6.4f}_seed{}_n{}.rle'.format(args.results,l,args.prefill,args.keep,args.seed,n))
        lmax=l
        if k >= (patience/2):
            patience *= 2
        k=1
    elif l==lmax:
        if random.random()>keep: # keep some of the "harmless" mutations
            for (x,y) in xy:
                pat[x,y] ^= 1 # revert
    else:
        for (x,y) in xy:
            pat[x,y] ^= 1 # revert

    if args.verbose and n%1000==0:
        log('',n,k,l,m,pat.population,d,r,patience,keep)

    if k>patience: # reset if stuck
        l = lifespan(pat) # recompute
        log('FINAL',n,k,l,m,pat.population,d,r,patience,keep)
        pat.centre().save('{}/final_L{:09d}_P{:04d}_K{:6.4f}_seed{}_n{}.rle'.format(args.results,l,args.prefill,args.keep,args.seed,n))
        exit()
