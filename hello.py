import argparse
import random
import datetime
import numpy as np ; print('numpy ' + np.__version__)
import lifelib ; print('lifelib',lifelib.__version__)

np.set_printoptions(linewidth=250)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--results', help='results directory', default='./results')
parser.add_argument('--seed', help='random seed', default=None, type=int)
parser.add_argument('--memory', help='garbage collection limit in MB', default=30000, type=int)
parser.add_argument('--summary',help='only save final pattern',default=False, action='store_true')
parser.add_argument('--verbose', default=False, action='store_true')
args = parser.parse_args()
if args.seed is None:
    args.seed = random.randint(1,1000000)
random.seed(args.seed)
np.random.seed(args.seed)
print(args)

def log(hdr,n,k,l,m,pop,d,r,patience,keep,advance):
    print('{:10} wall {} k {:6d} n {:6d} \033[1mLIFE\033[0m {:6d} pop {:6d} m {:6d} r {:12.8f} density {:12.8f} patience {:12.0f} keep {:12.8f} advance {:6d}'.format(hdr,datetime.datetime.now(),k,n,l,pop,m,r,d,patience,keep,advance))

# run soup until population is stable
def lifespan(pat,advance):
        last=0
        itot=0
        for i in range(1000000//advance):
            pat = pat.advance(advance+(i%7))
            itot += advance+(i%7)
            if pat.population == last:
                return itot
            else:
                last=pat.population
        return -1

sess = lifelib.load_rules("b3s23")
lt = sess.lifetree(memory=args.memory) # 50GB RAM
pat = lt.pattern() # empty pattern

nrun=0
lmax=1
n=0
k=0
pat[0,0] = 1 # seed

while True:
    n+=1
    k+=1
    patience = lmax
    advance = 2**int(np.log(lmax))

    # apply random mutations
    d = pat.population / (pat.bounding_box[2]*pat.bounding_box[3]) # density
    r = np.sqrt(pat.population) # radius
    p=1/r
    keep = -p*np.log(p)
    m = random.expovariate(1)
    m = int(np.ceil(m))
    xy=[(int(random.normalvariate(0,r)),int(random.normalvariate(0,r))) for k in range(m)]
    for (x,y) in xy:
        pat[x,y] ^= 1
    # early mutations can result in emptiness
    if pat.bounding_box is None:
        pat[0,0] = 1 # seed
        continue

    # use lifelib to compute lifespan
    l = lifespan(pat,advance)

    if l<0: # RUNAWAY
        log('RUNAWAY',n,k,l,m,pat.population,d,r,patience,keep,advance)
        pat.centre().save('{}/runaway_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header=None, footer=None, comments=str(args), file_format='rle', save_comments=True)
        for (x,y) in xy:
            pat[x,y] ^= 1 # revert
        nrun+=1
        if nrun>100:
            break
    elif l>lmax:
        log('BEST',n,k,l,m,pat.population,d,r,patience,keep,advance)
        if not args.summary:
            pat.centre().write_rle('{}/best_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header=None, footer=None, comments=str(args), file_format='rle', save_comments=True)
        lmax=l
        k=1
    elif l==lmax:
        if random.random()>keep: # keep some of the "harmless" mutations
            for (x,y) in xy:
                pat[x,y] ^= 1 # revert
    else:
        for (x,y) in xy:
            pat[x,y] ^= 1 # revert

    if args.verbose and n%1000==0:
        log('',n,k,l,m,pat.population,d,r,patience,keep,advance)

    if k>patience: # reset if stuck
        l = lifespan(pat,advance) # recompute
        log('FINAL',n,k,l,m,pat.population,d,r,patience,keep,advance)
        if l>1:
            pat.centre().save('{}/final_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header=None, footer=None, comments=str(args), file_format='rle', save_comments=True)
        break
