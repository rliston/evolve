import argparse
import random
import datetime
import time
import numpy as np ; print('numpy ' + np.__version__)
import lifelib ; print('lifelib',lifelib.__version__)
import scipy.stats
import os
import copy
import glob
import re
#from treelib import Node, Tree
#import pickle

np.set_printoptions(linewidth=250)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--p16', default=False, action='store_true')
parser.add_argument('--figeight', default=False, action='store_true')
parser.add_argument('--blank', default=False, action='store_true')
parser.add_argument('--traffic', default=False, action='store_true')
parser.add_argument('--weighted', default=False, action='store_true')
parser.add_argument('--delete', default=False, action='store_true')
parser.add_argument('--ttet', default=False, action='store_true')
parser.add_argument('--tub', default=False, action='store_true')
parser.add_argument('--pihep', default=False, action='store_true')
parser.add_argument('--squeeze', help='cmean coefficient', default=1, type=float)
parser.add_argument('--bipolar', default=16, type=int)
#parser.add_argument('--gaussian', default=None, type=int)
parser.add_argument('--xrad', help='gaussian radius', default=1, type=float)
parser.add_argument('--yrad', help='gaussian radius', default=1, type=float)
#parser.add_argument('--keepout', help='blank area at origin for spark', default=1, type=int)
parser.add_argument('--spark', default=False, action='store_true')
parser.add_argument('--nspark', help='number of initial sparks in tree', default=1, type=int)
parser.add_argument('--ispark', help='number of sparks in each nspark', default=1, type=int)
parser.add_argument('--mutalgo', help='ternary, remove', default='ternary')
parser.add_argument('--checkpoint', help='log interval', default=10000000000, type=int)
parser.add_argument('--space', help='grid spacing', default=3, type=int)
parser.add_argument('--meth', default=False, action='store_true')
parser.add_argument('--rpent', default=False, action='store_true')
parser.add_argument('--glider', default=False, action='store_true')
parser.add_argument('--bigblock', default=False, action='store_true')
parser.add_argument('--blinker', default=False, action='store_true')
parser.add_argument('--block', default=False, action='store_true')
#parser.add_argument('--states', help='glider_block_blinker', default='const')
parser.add_argument('--radalgo', help='radius adjustment', default='const')
#parser.add_argument('--block', default=False, action='store_true')
parser.add_argument('--hysteresis', help='prune fraction', default=0.5, type=float)
parser.add_argument('--best', default=False, action='store_true')
parser.add_argument('--irad', help='initial radius', default=10, type=int)
parser.add_argument('--fade', help='mutate delete glider rate', default=0.5, type=float)
parser.add_argument('--nullstate', default=False, action='store_true')
parser.add_argument('--fast', default=False, action='store_true')
#parser.add_argument('--sample', help='sampling pdf', default='uniform')
parser.add_argument('--init', help='initial lattice', default='grid')
parser.add_argument('--mutfreq', help='mutation frequency {single,expo}', default='expo')
parser.add_argument('--linit', help='minimum lifespan for initial seeds', default=0, type=int)
parser.add_argument('--mutset', help='mutation disjoint sets {single,dual}', default='single')
parser.add_argument('--dup', default=False, action='store_true')
parser.add_argument('--reseed', default=False, action='store_true')
parser.add_argument('--ipool', help='minimum samples before pruning', default=100, type=int)
#parser.add_argument('--cmax', help='prune threshold', default=4, type=float)
parser.add_argument('--prune', help='prune threshold', default=20, type=float)
#parser.add_argument('--pmax', help='prune threshold', default=200, type=float)
#parser.add_argument('--pool', help='prune to mean if tree exceeds args.pool length', default=100, type=int)
parser.add_argument('--rad', help='gaussian radius', default=0, type=float)
parser.add_argument('--n', help='gaussian gliders', default=10, type=int)
parser.add_argument('--spacex', help='grid spacing', default=10, type=int)
parser.add_argument('--spacey', help='grid spacing', default=10, type=int)
parser.add_argument('--sidex', help='grid size = side*2+1', default=50, type=int)
parser.add_argument('--sidey', help='grid size = side*2+1', default=50, type=int)
parser.add_argument('--seed', help='random seed', default=None, type=int)
parser.add_argument('--memory', help='garbage collection limit in MB', default=50000, type=int)
parser.add_argument('--results', help='results directory', default='./results')
parser.add_argument('--rleinit', help='directory of .rle files to seed the initial pool (replaces random spark init)', default=None)
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
if args.seed is None:
    args.seed = random.randint(1,1000000)
random.seed(args.seed)
np.random.seed(args.seed)
print(args)

sess = lifelib.load_rules("b3s23")
lt = sess.lifetree(memory=args.memory)

#current_time=`$(date "+%Y.%m.%d-%H.%M.%S")`
if not os.path.exists('results'):
    os.makedirs('results')
logf = open('results/log.{}'.format(datetime.datetime.now().isoformat()), 'w')
print('ARGS', args, file=logf)

def log(hdr):
    print('{:10} wall {} n {:6d} k {:6d} LIFE {:9.0f} ath {:9.0f} pop {:4d} m {:4d} prune {:8.0f} node {:6d} final {:6d} lprev {:6.0f} cmax {:6.0f} lmean {:9.0f} uniq {:9.0f} lmax {:9.0f} cmean {:6.2f} lmin {:9.3f} rad {:6.2f}'.format(
        hdr,datetime.datetime.now(),n,n1,l,ath,pat.population,m,prune,len(tree),pop,lprev,np.max(carr),np.mean(larr),len(uniq),np.max(larr),np.mean(carr),np.min(larr),rad))
    print('{:10} wall {} n {:6d} k {:6d} LIFE {:9.0f} ath {:9.0f} pop {:4d} m {:4d} prune {:8.0f} node {:6d} final {:6d} lprev {:6.0f} cmax {:6.0f} lmean {:9.0f} uniq {:9.0f} lmax {:9.0f} cmean {:6.2f} lmin {:9.3f} rad {:6.2f}'.format(
        hdr,datetime.datetime.now(),n,n1,l,ath,pat.population,m,prune,len(tree),pop,lprev,np.max(carr),np.mean(larr),len(uniq),np.max(larr),np.mean(carr),np.min(larr),rad),file=logf)
    logf.flush()

# run soup until population is stable, starting at generation lmax
def lifespan(pat,period=100,lmax=0,step=1):
        o = int(max(0,lmax-period))
        pat = pat.advance(o) # jump to lmax
        threshold = 0.8*np.log(period)
        for j in range(150000//(step*period)):
            pat,pt = pat.advance_trace(step,period)
            value,counts = np.unique(pt, return_counts=True) # histogram of population trace
            e = scipy.stats.entropy(counts,base=None) # entropy of population distribution
            if e<threshold: # HEURISTIC: threshold
                return max(0,o+j*step*period),pat.population
        return -1,pat.population # runaway

#transforms = ["flip","rot180","identity","transpose","flip_x","flip_y","rot90","rot270","swap_xy","swap_xy_flip","rcw","rccw"]
# 2 phases X 4 rotations
def transform_coords(coords, N=9):
    """
    Transforms a list of (x, y) coordinates using 12 different geometric operations.
    Assumes a grid range from 0 to N (inclusive).
    """
    # Define the 12 transformations as lambda functions
    transform_map = {
        "identity":     lambda x, y: (x, y),
        "flip":         lambda x, y: (N - x, y),            # Horizontal flip (mirror x)
        "flip_x":       lambda x, y: (N - x, y),            # Same as horizontal flip
        "flip_y":       lambda x, y: (x, N - y),            # Vertical flip (mirror y)
        "transpose":    lambda x, y: (y, x),                # Swap x and y
        "swap_xy":      lambda x, y: (y, x),                # Same as transpose
        "rot90":        lambda x, y: (y, N - x),            # 90° Clockwise
        "rot180":       lambda x, y: (N - x, N - y),        # 180° Rotation
        "rot270":       lambda x, y: (N - y, x),            # 270° Clockwise
        "rcw":          lambda x, y: (y, N - x),            # Rotate Clockwise (90°)
        "rccw":         lambda x, y: (N - y, x),            # Rotate Counter-Clockwise (90°)
        "swap_xy_flip": lambda x, y: (N - y, N - x)         # Transpose across anti-diagonal
    }
    results = {}
    for name, func in transform_map.items():
        results[name] = [func(x, y) for x, y in coords]
    return results

gliders=[]
if args.p16:
    on_cells = [ (0, 5), (0, 6), (1, 2), (1, 5), (1, 6), (2, 1), (2, 6), (3, 1), (3, 6), (4, 6), (5, 0), (5, 1), (5, 3), (5, 5), (6, 0), (6, 1), (6, 3), (6, 4), (8, 5), (8, 6), (8, 7) ]
    coords = transform_coords(on_cells)
    for t in coords:
        gliders.append(coords[t])
    #gliders.append(np.array([[None,None]])) # intentional blank

if args.blank:
    for _ in range(1):
        gliders.append(np.array([[None,None]])) # intentional blank

if args.glider:
    gliders.append(np.array([[0,-1],[1,-1],[-1,0],[0,0],[1,1]]))
    gliders.append(np.array([[-1,-1],[0,0],[1,0],[-1,1],[0,1]]))
    gliders.append(np.array([[0,-1],[0,0],[1,0],[-1,1],[1,1]]))
    gliders.append(np.array([[-1,-1],[1,-1],[-1,0],[0,0],[0,1]]))
    gliders.append(np.array([[-1,-1],[0,-1],[-1,0],[1,0],[-1,1]]))
    gliders.append(np.array([[1,-1],[-1,0],[1,0],[0,1],[1,1]]))
    gliders.append(np.array([[-1,-1],[0,-1],[1,-1],[1,0],[0,1]]))
    gliders.append(np.array([[0,-1],[-1,0],[-1,1],[0,1],[1,1]]))

if args.traffic:
    for _ in range(1):
        gliders.append(np.array([[-3,-1],[-3,0],[-3,1],[-1,3],[0,3],[1,3],[3,-1],[3,0],[3,1],[-1,-3],[0,-3],[1,-3]]))

if args.figeight:
    three = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]
    gliders.append(np.concatenate([three, [(3+x,3+y) for (x,y) in three]], axis=0))
    gliders.append(np.concatenate([[(3+x,0+y) for (x,y) in three], [(0+x,3+y) for (x,y) in three]], axis=0))
    gliders.append(np.array([[None,None]])) # intentional blank
#    for _ in range(4):
#        gliders.append(np.concatenate([three, [(3+x,3+y) for (x,y) in three]], axis=0))
#        gliders.append(np.concatenate([[(3+x,0+y) for (x,y) in three], [(0+x,3+y) for (x,y) in three]], axis=0))
#        #gliders.append(np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]])) # 3x3 block

if args.bigblock:
    for _ in range(8):
        gliders.append(np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]])) # 3x3 block

if args.blinker:
    for _ in range(4):
        gliders.append(np.array([[0,-1],[0,0],[0,1]])) # blinker
        gliders.append(np.array([[-1,0],[0,0],[1,0]])) # blinker

if args.block:
    for _ in range(2):
        gliders.append(np.array([[0,0],[0,1],[1,0],[1,1]])) # block
        gliders.append(np.array([[-1,0],[-1,1],[0,0],[0,1]])) # block
        gliders.append(np.array([[0,-1],[0,0],[1,-1],[1,0]])) # block
        gliders.append(np.array([[-1,-1],[-1,0],[0,-1],[0,0]])) # block

if args.rpent:
    gliders.append(np.array([[0, -1], [0, 0], [0, 1], [-1, 0], [1, 1]]))
    gliders.append(np.array([[-1, 0], [0, 0], [1, 0], [0, 1], [1, -1]]))
    gliders.append(np.array([[0, 1], [0, 0], [0, -1], [1, 0], [-1, -1]]))
    gliders.append(np.array([[1, 0], [0, 0], [-1, 0], [0, -1], [-1, 1]]))
    gliders.append(np.array([[0, 1], [0, 0], [0, -1], [-1, 0], [1, -1]]))
    gliders.append(np.array([[0, -1], [0, 0], [0, 1], [1, 0], [-1, 1]]))
    gliders.append(np.array([[-1, 0], [0, 0], [1, 0], [0, -1], [1, 1]]))
    gliders.append(np.array([[1, 0], [0, 0], [-1, 0], [0, 1], [-1, -1]]))

if args.meth:
    coords = np.array([[-1,0],[1,0],[0,1],[-1,1],[-2,-1], [1,-1],[2,-1]])
    # Identity (0-degree rotation)
    gliders.append(coords)
    # 90-degree clockwise rotation
    gliders.append(np.array([[y, -x] for x, y in coords]))
    # 180-degree rotation
    gliders.append(np.array([[-x, -y] for x, y in coords]))
    # 270-degree clockwise rotation (or 90-degree counter-clockwise)
    gliders.append(np.array([[-y, x] for x, y in coords]))
    # Reflection across the x-axis
    gliders.append(np.array([[x, -y] for x, y in coords]))
    # Reflection across the y-axis
    gliders.append(np.array([[-x, y] for x, y in coords]))
    # Reflection across the line y = x
    gliders.append(np.array([[y, x] for x, y in coords]))
    # Reflection across the line y = -x
    gliders.append(np.array([[-y, -x] for x, y in coords]))

if args.pihep:
    coords = np.array([[-1,-1],[-1,0],[-1,1],[0,1],[1,-1],[1,0],[1,1]])
    gliders.append(coords)
    gliders.append(np.array([[y, -x] for x, y in coords]))
    gliders.append(np.array([[-x, -y] for x, y in coords]))
    gliders.append(np.array([[-y, x] for x, y in coords]))
    gliders.append(np.array([[x, -y] for x, y in coords]))
    gliders.append(np.array([[-x, y] for x, y in coords]))
    gliders.append(np.array([[y, x] for x, y in coords]))
    gliders.append(np.array([[-y, -x] for x, y in coords]))

if args.tub:
    coords = np.array([[0,1],[0,-1],[-1,0],[1,0]])
    gliders.append(coords)
    gliders.append(np.array([[y, -x] for x, y in coords]))
    gliders.append(np.array([[-x, -y] for x, y in coords]))
    gliders.append(np.array([[-y, x] for x, y in coords]))
    gliders.append(np.array([[x, -y] for x, y in coords]))
    gliders.append(np.array([[-x, y] for x, y in coords]))
    gliders.append(np.array([[y, x] for x, y in coords]))
    gliders.append(np.array([[-y, -x] for x, y in coords]))

if args.ttet:
    coords = np.array([[-1,0],[0,0],[1,0],[0,-1]])
    gliders.append(coords)
    gliders.append(np.array([[y, -x] for x, y in coords]))
    gliders.append(np.array([[-x, -y] for x, y in coords]))
    gliders.append(np.array([[-y, x] for x, y in coords]))
    gliders.append(np.array([[x, -y] for x, y in coords]))
    gliders.append(np.array([[-x, y] for x, y in coords]))
    gliders.append(np.array([[y, x] for x, y in coords]))
    gliders.append(np.array([[-y, -x] for x, y in coords]))


#states=[0,1,2,3,4,5,6,7,8,9,10,11] # 8 glider, 2 blinker, 1 block, 1 tub
states = range(len(gliders))

#spark=np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]])
#spark=np.array([[0,-1],[0,0],[0,1],[-1,0],[1,1]])
#spark_rle = '''
#6bobob$5bo4b$2o2bo4bo$2obo2bob2o$4b2o!
#'''
#spark_pat = lt.pattern(spark_rle)

def render(pat,v):
#   pat[-100:100,-100:100]=0
    bb = pat.bounding_box
    if bb is not None:
        pat[bb[0]:bb[0]+bb[2],bb[1]:bb[1]+bb[3]]=0

#    for (x0,y0,s) in prev:
#        #x0 = int(np.around(x0))
#        #y0 = int(np.around(y0))
#        xy = [[x0+x,y0+y] for x in range(-1,2) for y in range(-1,2)]
#        xy = np.array(xy)
#        pat[xy]&=0

#    if args.spark:
#        pat[spark] |=1
#        #pat |= spark_pat
#    #print('v',v)
    # v is either a plain dict {(gx,gy): state} (random-init)
    # or a (base_rle_pat, vec_dict) tuple (--rleinit)
    if isinstance(v, tuple):
        base_rle, vec = v
        rle_bb = base_rle.bounding_box
        if rle_bb is not None:
            rx, ry, rw, rh = rle_bb
            pat[rx:rx+rw, ry:ry+rh] |= base_rle[rx:rx+rw, ry:ry+rh]
    else:
        vec = v
    for key in vec.keys():
        s = vec[key]
        #print('key', key, 's',s)
        x0 = key[0]*args.spacex
        y0 = key[1]*args.spacey
        xy = [[x0+x,y0+y] for (x,y) in gliders[s] if x is not None]
        xy = np.array(xy)
        pat[xy]|=1
#    if pat.population==13:
#        print('v',v)
#        print('bb',bb)
#        fn = '{}/debug_seed{:09d}.rle'.format(args.results,args.seed)
#        bb = pat.bounding_box
#        pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
#        exit()

tree=[]
uniq=[]
pat = lt.pattern()
#gvec = [(x,y) for x in range(-args.sidex//2,args.sidex//2,args.spacex) for y in range(-args.sidey//2,args.sidey//2,args.spacey)]

# INITIAL PATTERNS
#gaussian_svec = [(int(np.random.normal(0,args.rad)),int(np.random.normal(0,args.rad)),None) for i in range(args.n)] # global
#def append_tree():
#    pat = lt.pattern()
#    while True:
#        if args.init=='grow':
#            svec=[(0,0,None)]
#        if args.init=='fill':
#            svec=[]
#            f=[x for x in states]+[None]
#            for x in range(-args.sidex//2,args.sidex//2,args.spacex):
#                for y in range(-args.sidey//2,args.sidey//2,args.spacey):
#                    svec.append((x,y,random.choice(f)))
#        elif args.init=='fill2':
#            svec=[]
#            for x in range(-args.sidex//2,args.sidex//2,args.spacex):
#                for y in range(-args.sidey//2,args.sidey//2,args.spacey):
#                    if x%2 and y%2:
#                        svec.append((x,y,random.choice(states)))
#                    else:
#                        svec.append((x,y,None))
#        elif args.init=='grid':
#            svec=[]
#            for _ in range(args.n):
#                (x,y) = random.choice(gvec)
#                svec.append((x,y,random.choice(states)))
#        elif args.init=='sweep':
#            svec = [(int(x),int(y),random.choice(states)) for x in np.geomspace(1,args.sidex,num=args.spacex) for y in np.geomspace(1,args.sidey,num=args.spacey)]
#        elif args.init=='gaussian':
#            svec=[]
#            for _ in range(args.n):
#                while True:
#                    x = (np.random.normal(0,args.irad) // args.space) * args.space
#                    y = (np.random.normal(0,args.irad) // args.space) * args.space
#                    allxy = [(z[0],z[1]) for z in svec]
#                    if (x,y) not in allxy:
#                        svec.append((x,y,random.choice(states)))
#                        break
#            #gaussian_svec = [(x,y,None) for i in range(args.n)]
#            #svec = [(x,y,random.choice(states)) for (x,y,_) in gaussian_svec]
#        elif args.init=='flat':
#            flat_svec = [(np.random.randint(-args.irad,args.irad),0,None) for i in range(args.n)]
#            svec = [(x,y,random.choice(states)) for (x,y,_) in flat_svec]
#
#        render(pat,svec)
#        if args.linit>0: # linit is the minimum lifespan for init patterns
#            l,pop = lifespan(pat)
#            if l>args.linit:
#                    break
#        else:
#            l=0
#            pop=pat.population
#            break
#    # mut
#    #vlen = len(svec)
#    #mset = list(range(0,vlen))
#    #random.shuffle(mset)
#    #mset1=mset[0:vlen//2]
#    #mset2=mset[vlen//2:]
#    #tree.append([l,svec,0,(mset1,mset2)])
#    tree.append([l,svec,0,None])
#    #uniq[pat.digest()]=True
#    uniq.append(True)
#    return pat,l,pop,svec
#
#for k in range(args.ipool):
#    pat,l,pop,svec = append_tree()
#    print('svec',k,l,pop,len(svec))
#    #if k==0:
#    if True:
#        fn = '{}/init_K{:06d}_seed{:09d}.rle'.format(args.results,k,args.seed)
#        bb = pat.bounding_box
#        pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)

#svec = [None]

if args.rleinit:
    # Seed the pool from all .rle files in the given directory
    rle_files = sorted(f for f in glob.glob(os.path.join(args.rleinit, '*.rle'))
                       if 'runaway' not in os.path.basename(f))
    print('RLEINIT: loading {} files from {}'.format(len(rle_files), args.rleinit))
    for k, fn in enumerate(rle_files):
        with open(fn, 'r') as f:
            rle_text = f.read()
        rle_pat = lt.pattern(rle_text)
        # Extract encoded lifespan from filename (e.g. BEST_P001234_L047300_seed...)
        # and use as lmax hint so lifespan() can HashLife-jump near the stable window
        # instead of advancing one generation at a time from gen 0.
        m = re.search(r'_L(\d+)_', os.path.basename(fn))
        lhint = int(m.group(1)) if m else 0
        l, final = lifespan(rle_pat, lmax=lhint)
        print('RLEINIT k {:4d} fn {} lhint {} l {} pop {} final {}'.format(
            k, os.path.basename(fn), lhint, l, rle_pat.population, final))
        if l > 0:
            tree.append([l, (rle_pat, {}), 0])
    print('RLEINIT: {} patterns added to tree'.format(len(tree)))
else:
    for k in range(args.nspark):
#    vec[(0,0)] = random.choice(states)
#    for x in range(-int(args.irad), int(args.irad)):
#        vec[(x,0)] = random.choice(states)
#
        while True:
            vec={}
            for _ in range(1+np.random.randint(args.ispark)):
                #vec[(np.floor(np.random.normal(0,args.irad)),np.floor(np.random.normal(0,args.irad)))] = random.choice(states)
                vec[(np.floor(np.random.uniform(-args.irad,args.irad)),np.floor(np.random.uniform(-args.irad,args.irad)))] = random.choice(states)
            render(pat,vec)
            l,final = lifespan(pat)
            if l==0:
                continue
            bb = pat.bounding_box
            if k<10 and bb is not None:
                fn = '{}/init_K{:06d}_seed{:09d}.rle'.format(args.results,k,args.seed)
                pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
            print('INIT','k',k,'l',l,'pop',pat.population,'final',final)
            tree.append([l,vec,0])
            break
            #tree.append([args.linit,vec,0])
#    if args.bipolar is None:
#        tree.append([args.linit,{(0,0):random.choice(states)},0]) # lifespan, pat[(x,y)] = s
#    else:
#        tree.append([args.linit,{(-args.bipolar,0):random.choice(states), (args.bipolar,0):random.choice(states)},0]) # lifespan, pat[(x,y)] = s
'''
for k in range(args.nspark):
    while True:
        vec={}
        for x in range(-args.irad, args.irad):
            for y in range(-args.irad, args.irad):
                vec[x,y] = random.choice(states)
        render(pat,vec)
        l,final = lifespan(pat)
        if l==0:
            continue
        bb = pat.bounding_box
        if k<10 and bb is not None:
            fn = '{}/init_K{:06d}_seed{:09d}.rle'.format(args.results,k,args.seed)
            pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
        print('INIT','k',k,'l',l,'pop',pat.population,'final',final)
        tree.append([l,vec,0])
        break
'''

# MAIN LOOP
n=0
n0=0
n1=0
#ndup=0
ath=0
lmax0=None
m=0
prune = args.prune
pat=lt.pattern()
nbest=0
#rad = args.rad
ntime=time.time()
try:
    while True:
        if len(tree)==0:
            print('EMPTY TREE')
            break
        # statistics
        larr = np.array([x[0] for x in tree])
        carr = np.array([x[2] for x in tree])

        if args.radalgo=='const':
            rad = args.rad
        elif args.radalgo=='pi':
            rad = np.sqrt(np.mean(larr))/np.pi
            rad /= args.spacex

        #if args.debug and np.mean(carr)>50:
        #    exit()
        #rad = 1+np.mean(carr)
        #rad = 1+np.sqrt(np.mean(carr))
    #    if args.radalgo=='grid':
    #        rad = np.mean(carr)
    #        rad = np.sqrt(rad)
    #        rad = args.space+int(rad)
    #        rad += rad%2
    #        rad *= args.space
    #        #rad = max(rad,args.irad)
    #        #rad = np.sqrt(np.mean(larr))/np.pi
    #        #rad = int(rad)
    #        #rad += rad%2 # make rad even
    #        #rad = rad*args.spacex
    #        args.sidex = rad
    #        args.sidey = rad
    #        gvec = [(x,y) for x in range(-args.sidex//2,args.sidex//2,args.spacex) for y in range(-args.sidey//2,args.sidey//2,args.spacey)]
    #    if args.radalgo=='pi':
    #        rad = np.sqrt(np.mean(larr))/np.pi
    #        rad *= 0.5
    #    if args.radalgo=='cmean':
    #        rad = np.mean(carr)
    #        rad = max(rad,args.irad)
    #    if args.radalgo=='sqrt':
    #        rad = 2*np.sqrt(np.mean(carr))
        #rad = np.mean(carr) + args.irad
        #cmax = args.cmax/np.log(len(tree))
        #cmax = args.cmax
        #log('DEBUG')
        #pool=args.pool
        #if np.std(carr)>args.pool:
        #    pool = args.pool*np.mean(carr)
        #else:
        #    pool = args.pool
    
        # save interesting patterns
        if lmax0 is None:
            lmax0 = np.max(larr)
        if (np.max(larr)!=lmax0):
            lmax0=np.max(larr)
            #pat0=lt.pattern()
            ref = tree[np.argmax(larr)][1]
            render(pat,ref)
            bb = pat.bounding_box
            if bb is not None:
                fn = '{}/lmax_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,pat.population,int(lmax0),args.seed,n)
                pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
            log('LMAX')
    
        # sample the pool of initial state vectors
        if args.weighted:
            if np.sum(larr)==0:
                nodeidx = np.random.choice(np.arange(0,len(tree)))
            else:
                prob = larr/np.sum(larr)
                nodeidx = np.random.choice(np.arange(0,len(tree)), p=prob)
        else:
            nodeidx = np.random.choice(np.arange(0,len(tree)))
        #nodeidx = np.random.randint(0,len(tree))
    #    if args.sample == 'weighted':
    #        prob = np.log(1+larr)
    #        prob = prob/np.sum(prob)
    #        nodeidx = np.random.choice(np.arange(0,len(tree)),p=prob)
    #    elif args.sample == 'uniform':
    #        nodeidx = np.random.choice(np.arange(0,len(tree)))
    
        (lmax,ivec,nc) = tree[nodeidx]
        #rad = args.rad+nc
        #rad = args.rad
        #rad = np.sqrt(nc)
    
        # mutate
        # ivec is either a plain dict (random-init) or a (base_rle, vec) tuple (--rleinit)
        if isinstance(ivec, tuple):
            imut_base, imut_vec = ivec[0], ivec[1].copy()
        else:
            imut_base = None
            imut_vec = ivec.copy()
    #    if args.mutfreq=='single':
    #        m=1
    #    if args.mutfreq=='double':
    #        m=2
    #    if args.mutfreq=='triple':
    #        m=3
    #    if args.mutfreq=='expo':
    #        m = int(np.ceil(random.expovariate(1)))
    #    if args.mutfreq=='expoplus':
    #        m = 1+int(np.ceil(random.expovariate(1)))
    #    if args.mutfreq=='uniform':
    #        m = random.choice([1,2,3])
        #m = np.random.randint(1,len(ivec))
        #m = int(np.ceil(random.expovariate(1/(1+np.max(carr)))))
        #m = 1+np.random.randint(1+np.ceil(np.mean(carr)))
        #m = int(len(ivec)/(1+np.mean(carr)))
        #print('m',m)
    
        m = int(np.ceil(random.expovariate(1)))
        #m = 1
        #rad = 0.5*np.sqrt(len(uniq)) # grow radius each generation
        #rad = np.log(1+len(uniq)) # grow radius each generation
        for i in range(m):
            if args.delete:
                action = random.choice(['add','del'])
            else:
                action = 'add'
            # for RLE-base patterns the base provides cells even with empty vec,
            # so allow deleting the last tile; for plain-dict mode keep at least 1
            min_vec = 0 if imut_base is not None else 1
            if action=='del' and len(imut_vec)>min_vec:
                key = random.choice(list(imut_vec.keys()))
                del imut_vec[key]
            else:
                dx = np.random.normal(0,rad)
                dy = np.random.normal(0,rad)
                #dx = np.random.uniform(-args.irad,args.irad)
                #dy = np.random.uniform(-args.irad,args.irad)

                #key = random.choice(list(imut_vec.keys()))
                #dx = random.choice([-int(rad),-1,0,1,int(rad)])
                #dy = random.choice([-int(rad),-1,0,1,int(rad)])
                #dx = random.choice([-31,-15,-7,-3,-1,0,1,3,7,15,31])
                #dy = random.choice([-31,-15,-7,-3,-1,0,1,3,7,15,31])
                #dx = random.choice([-args.bipolar,-1,1,args.bipolar])
                #dy = random.choice([-args.bipolar,-1,1,args.bipolar])
                #dx = random.choice([-args.bipolar,args.bipolar])
                #dy = random.choice([-args.bipolar,args.bipolar])

                #imut_vec[(int(key[0]+dx),int(key[1]+dy))] = random.choice(states)
                imut_vec[(int(dx),int(dy))] = random.choice(states)
    
    #        if len(imut)==0:
    #            mode='add'
    #            x0=0
    #            y0=0
    #            key=(x0,y0)
    #        else:
    #            key = random.choice(list(imut.keys()))
    #            #print('key',key,type(key),type(key[0]),type(key[1]))
    #            x0 = key[0]
    #            y0 = key[1]
    #            #mode = random.choice(['modify','add'])
    #            mode='add'
    #            #weights = [0.3, 0.6, 0.1]
    #            #weights = [0.4, 0.4, 0.2]
    #            #weights = [0.0, 0.5, 0.5]
    #            #weights = [0.1, 0.8, 0.1]
    #            #weights = [0.0, 1.0, 0.0]
    #            #mode = random.choices(['modify','add','delete'], weights=weights)[0]
    #
    #        if mode=='delete':
    #            del imut[key]
    #        elif mode=='modify':
    #            imut[key] = random.choice(states)
    #        elif mode=='add':
    #            #dx = args.spacex*np.random.randint(1, 1+rad)*random.choice([1,-1])
    #            #dy = args.spacey*np.random.randint(1, 1+rad)*random.choice([1,-1])
    #            #dx = random.choice([1,-1])
    #            #dy = random.choice([1,-1])
    #            dx = np.floor(np.random.normal(0,args.xrad))
    #            dy = np.floor(np.random.normal(0,args.yrad))
    #            #x0=0
    #            #y0=0
    #            #y0 = random.choice([-1,0,1])
    #            #dy=0
    #            #dx=0
    #            #dy=0
    #            #x0 = np.round(np.random.normal(0,rad))
    #            #y0 = np.round(np.random.normal(0,rad))
    #            #if key==(int(x0+dx),int(y0+dy)):
    #            #    print('MODIFY',key)
    #            imut[(int(x0+dx),int(y0+dy))] = random.choice(states)
    #            #print('dx',dx,'dy',dy,'x0',x0,'y0',y0,'imut',imut)
    #            #print('imut',imut)
    
    
        #print('imut',imut)
    #        idx = np.random.randint(len(imut))
    #        if imut[idx][2] is None: # origin spark
    #            dx = imut[idx][0] + random.choice([-1,1])*args.spacex
    #            dy = imut[idx][1] + random.choice([-1,1])*args.spacey
    #        if args.mutalgo=='ternary':
    #            if imut[idx][2] is None:
    #                imut[idx] = (imut[idx][0],imut[idx][1],random.choice([0,1]))
    #            elif imut[idx][2] == 0:
    #                imut[idx] = (imut[idx][0],imut[idx][1],random.choice([None,1]))
    #            elif imut[idx][2] == 1:
    #                imut[idx] = (imut[idx][0],imut[idx][1],random.choice([None,0]))
    #        elif args.mutalgo=='remove':
    #            imut[idx] = (imut[idx][0],imut[idx][1],None)
    #
    
    #        if len(imut) > 0 and np.random.uniform() < args.fade:
    #            imut[idx] = (imut[idx][0],imut[idx][1],None)
    #        elif imut[idx][2] is None:
    #            imut[idx] = (imut[idx][0],imut[idx][1],random.choice([0,1]))
    #        elif imut[idx][2] == 0:
    #            imut[idx] = (imut[idx][0],imut[idx][1],1)
    #        else:
    #            imut[idx] = (imut[idx][0],imut[idx][1],0)
    #        if len(imut) > 0 and np.random.uniform() < args.fade:
    #            imut[idx] = (imut[idx][0],imut[idx][1],None)
    #            #del imut[idx]
    #        else:
    #            if imut[idx][2] is None:
    #                imut[idx] = (imut[idx][0],imut[idx][1],random.choice(states))
    #            elif imut[idx][2] == 0:
    #                imut[idx] = (imut[idx][0],imut[idx][1],1)
    #            else:
    #                imut[idx] = (imut[idx][0],imut[idx][1],0)
                #imut[idx][2] = random.choice(states)
                #(x,y) = random.choice(gvec)
    #            #imut.append((int(np.random.normal(0,rad)),int(np.random.normal(0,rad)),random.choice(states)))
    #            allxy = [(z[0],z[1]) for z in imut]
    #            while True:
    #                x = (np.random.normal(0,rad) // args.space) * args.space
    #                y = (np.random.normal(0,rad) // args.space) * args.space
    #                if (x,y) not in allxy:
    #                    imut.append((x,y,random.choice(states)))
    #                    break
    #        if args.mutset=='single':
    #            k = np.random.randint(0,len(imut))
    #            while True:
    #                if args.nullstate:
    #                    ns = random.choice(nullstates)
    #                else:
    #                    ns = random.choice(states)
    #                if ns!=imut[k][2]:
    #                    break
    #            imut[k] = [imut[k][0],imut[k][1],ns]
    #        elif args.mutset=='dual':
    #            #k = np.random.randint(0,len(imut)//2)*2 # even
    #            #k = np.random.randint(0,len(imut)//2)
    #            k = random.choice(mset1)
    #            while True:
    #                ns = random.choice(states)
    #                if ns!=imut[k][2]:
    #                    break
    #            imut[k] = [imut[k][0],imut[k][1],ns]
    #            #k = np.random.randint(0,len(imut)//2)*2+1 # odd
    #            #k = np.random.randint(len(imut)//2,len(imut))
    #            k = random.choice(mset2)
    #            while True:
    #                ns = random.choice(states)
    #                if ns!=imut[k][2]:
    #                    break
    #            imut[k] = [imut[k][0],imut[k][1],ns]
    
        # reconstruct imut in the appropriate format for render/storage
        imut = (imut_base, imut_vec) if imut_base is not None else imut_vec

        # run the pattern
        render(pat,imut)
    #    if (pat.population%5)!=0:
    #        print('pop',pat.population,'m',m,'bb',pat.bounding_box)
    #        print('imut',imut)
    #        print('allxy',allxy)
    #        bb = pat.bounding_box
    #        pat.write_rle('{}/debug_P{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,pat.population,args.seed,n), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
        #bb = pat.bounding_box
        #pat.write_rle('{}/snapshot.rle'.format(args.results), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
        l,pop = lifespan(pat)
        n+=1
    
        #if pat.digest() in uniq:
    #    if False:
    #        tree[nodeidx][2]+=1
    #        #log("DUP")
    #        ndup+=1
    #        #if args.dup:
    #        #    append_tree() # if dup then add new seed pattern to pool
        if l==-1:
            tree[nodeidx][2]+=1
            log('RUNAWAY')
            bb = pat.bounding_box
            if bb is not None:
                pat.write_rle('{}/runaway_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
        elif l>lmax:
        #elif l>=lmax:
            lprev = lmax
            n1=n-n0
            n0=n
            #uniq[pat.digest()]=True
            uniq.append(True)
            #tree[nodeidx][2]=0
            #tree[nodeidx][2] = tree[nodeidx][2] // 2
            #print('origin',origin)
            tree.append([l,imut,0])
            # prune
            #prune = args.squeeze*(args.prune+np.mean(carr))
            #prune = max(np.mean(carr), args.prune)
    
            while len(tree)>prune:
                tree.sort(key=lambda x: -x[2]) # PRUNE PATTERNS WITH MOST FAILED MUTATIONS
                tree = tree[1:]
                tree.sort(key=lambda x: x[0]) # PRUNE PATTERNS WITH LOWEST LIFESPAN
                tree = tree[1:]
    
    #            if random.choice([True, False]):
    #                tree.sort(key=lambda x: -x[2]) # PRUNE PATTERNS WITH MOST FAILED MUTATIONS
    #                tree = tree[1:]
    #            else:
    #                tree.sort(key=lambda x: x[0]) # PRUNE PATTERNS WITH LOWEST LIFESPAN
    #                tree = tree[1:]
    
            #cmax = np.log(np.mean(larr)) - np.log(1+len(tree))
            #tree = [x for x in tree if x[2]<cmax]
            #print([x for x in tree])
            #tree = [x for x in tree if x[2]<(x[3]-np.log(len(tree)))]
            #tree = [x for x in tree if x[2]<(0.5*x[3])]
            #ncmax=np.mean(carr)+np.std(carr)
            #if np.mean(carr)>args.cmax or len(tree)>args.pmax:
            #if len(tree)>args.pmax:
            #if (len(tree)>np.square(np.mean(carr))) and (len(tree)>args.prune):
            #if np.mean(carr)>args.prune:
            #if len(tree)>args.prune and min([x[2] for x in tree])>0:
            #if len(tree)>args.prune and np.mean(carr)>1:
    
    #        if len(tree)>prune:
    #            #tree = [x for x in tree if x[2]<np.max(carr)] # PRUNE PATTERNS WITH MOST FAILED MUTATIONS
    #            #tree = [x for x in tree if x[0]>np.min(larr)] # PRUNE PATTERNS WITH LOWEST LIFESPAN
    #            plim = int(np.ceil(prune*args.hysteresis))
    #            for p in range(plim):
    #                #tree.sort(key=lambda x: x[0]) # PRUNE PATTERNS WITH LOWEST LIFESPAN
    #                #tree = tree[1:]
    #                tree.sort(key=lambda x: -x[2]) # PRUNE PATTERNS WITH MOST FAILED MUTATIONS
    #                tree = tree[1:]
    
            #if args.reseed and len(tree)==1:
            #    log('SEED')
            #    append_tree() # if pop too low then add new seed pattern to pool
    
            #if len(tree)<=args.prune:
            #    log('SEED')
            #    append_tree() # add new seed pattern to pool
            #if len(tree)==0:
            #    print('EMPTY TREE')
            #    break
            # all time high
            if l>ath:
                ath = l
                bb = pat.bounding_box
                if bb is not None:
                    fn = '{}/ATH_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,pat.population,int(l),args.seed,n)
                    pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
                log('ATH')
            else:
                #if args.best and l>np.mean(larr):
                nbest +=1
                nt0 = time.time()
                if args.best or (nt0-ntime) > args.checkpoint:
                    #log('CHECK')
                    ntime=nt0
                    bb = pat.bounding_box
                    if bb is not None:
                        fn = '{}/BEST_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,pat.population,int(l),args.seed,n)
                        pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
                log('BEST')
    
        else:
            if action!='del':
                tree[nodeidx][2]+=1
    
    
except KeyboardInterrupt:
    print('\nSTOPPING MAIN')
    exit()
