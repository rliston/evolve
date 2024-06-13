import argparse
import random
import datetime
import time
import numpy as np ; print('numpy ' + np.__version__)
import lifelib ; print('lifelib',lifelib.__version__)
import scipy.stats
import os
import copy
#import treelib
from treelib import Node, Tree

np.set_printoptions(linewidth=250)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mininit', help='minimum init lifespan', default=1000, type=float)
parser.add_argument('--orient', default=False, action='store_true')
parser.add_argument('--blast', help='blast probability', default=1, type=float)
parser.add_argument('--sib', help='maximum nuber of siblings', default=1, type=float)
parser.add_argument('--exp', help='power law sampling exponent', default=3.0, type=float)
parser.add_argument('--nsprune', help='mean samples prune threshold', default=10, type=float)
parser.add_argument('--prune', default=False, action='store_true')
parser.add_argument('--mut', help='mutation alg {xor,ones}', default='xor')
parser.add_argument('--xorthresh', help='init density', default=0.5, type=float)
parser.add_argument('--gcspace', help='inner circle radius', default=0, type=int)
parser.add_argument('--gcinner', help='inner circle radius', default=0, type=int)
parser.add_argument('--gcrings', help='number of concentric circles', default=3, type=int)
parser.add_argument('--irad2', help='initial radius', default=10, type=float)
parser.add_argument('--gcoord', help='glider coordinate mapping {single,binary}', default='single')
parser.add_argument('--gvec', help='numer of perimiter glider positions', default=100, type=int)
parser.add_argument('--aspect', help='gaussian aspect ratio', default=1.00, type=float)
parser.add_argument('--tleaf', help='target number of leaf nodes', default=100, type=int)
parser.add_argument('--bot', help='prune threshold bottom perctentile', default=1.00, type=float)
parser.add_argument('--decay', help='sampling parameter', default=0.5, type=float)
parser.add_argument('--ptile', help='prune rate', default=1.0, type=float)
parser.add_argument('--pruneinit', help='prune init param', default=1, type=float)
parser.add_argument('--pow', help='sampling parameter', default=1.5, type=float)
parser.add_argument('--top', help='threshold for saving top patterns', default=99.73, type=float)
parser.add_argument('--rscale', help='r=sqrt(lmax)*rscale', default=1.0, type=float)
parser.add_argument('--under', help='under tleaf prune rate', default=1.0101010101, type=float)
parser.add_argument('--over', help='over tleaf prune rate', default=0.9900000000, type=float)
parser.add_argument('--rate', help='prune rate', default=0.01, type=float)
parser.add_argument('--srad', help='seed radius', default=10, type=float)
parser.add_argument('--nseed', help='number of radius 10 seed gliders', default=0, type=int)
parser.add_argument('--multimut', default=False, action='store_true')
parser.add_argument('--iroot', help='initial root count', default=1, type=int)
parser.add_argument('--inodes', help='initial node count', default=1, type=int)
parser.add_argument('--percentile', help='prune rate, 0-100', default=10, type=float)
parser.add_argument('--legalize', default=False, action='store_true')
parser.add_argument('--meanplus', default=False, action='store_true')
parser.add_argument('--height', help='manifold height', default=None, type=int)
parser.add_argument('--width', help='manifold width', default=None, type=int)
#parser.add_argument('--prune', help='prune rate', default=90, type=float)
parser.add_argument('--sigma', help='pruning parameter', default=1.0, type=float)
parser.add_argument('--sample', help='sampling parameter', default=0.5, type=float)
parser.add_argument('--step', help='random walk step size', default=1, type=float)
parser.add_argument('--mod', help='bound the INIT cursor brownian motion', default=None, type=int)
parser.add_argument('--n', help='init parameter', default=1, type=int)
parser.add_argument('--backtrack', help='backtrack threshold', default=1000000000, type=float)
parser.add_argument('--rr', help='root rate', default=100, type=int)
parser.add_argument('--rootlife', help='lifespan of root', default=1, type=int)
parser.add_argument('--alt5', default=False, action='store_true')
parser.add_argument('--alt4', default=False, action='store_true')
parser.add_argument('--alt3', default=False, action='store_true')
parser.add_argument('--alt2', default=False, action='store_true')
parser.add_argument('--alt', default=False, action='store_true')
parser.add_argument('--btboost', help='backtrack increase every lmax', default=1, type=float)
parser.add_argument('--binary', default=False, action='store_true')
parser.add_argument('--leaves', help='minimum pool size, else root', default=1, type=int)
parser.add_argument('--keepout', help='grid3 origin keepout square', default=100, type=int)
parser.add_argument('--orad', help='grid3 origin radius', default=0, type=float)
parser.add_argument('--minpool', help='minimum pool size, else root', default=1, type=int)
parser.add_argument('--growth', help='increase r when backtrack', default=1, type=float)
parser.add_argument('--lmax', help='force initial lmax when loading init rle', default=0, type=int)
parser.add_argument('--boom', default=False, action='store_true')
parser.add_argument('--keep', default=False, action='store_true')
parser.add_argument('--square', default=False, action='store_true')
parser.add_argument('--log', default=False, action='store_true')
parser.add_argument('--survive', help='surviving nodes after backtrack, in terms of std', default=1, type=float)
parser.add_argument('--showtree', default=False, action='store_true')
parser.add_argument('--topk', help='backtrack threshold', default=0, type=float)
parser.add_argument('--effort', help='backtrack threshold 0..1', default=1, type=float)
parser.add_argument('--nath', help='nath threshold for backtrack', default=10000, type=int)
parser.add_argument('--space', help='grid spacing', default=25, type=int)
parser.add_argument('--spacex', help='grid spacing', default=25, type=int)
parser.add_argument('--spacey', help='grid spacing', default=25, type=int)
parser.add_argument('--side', help='grid size = side*2+1', default=2, type=int)
parser.add_argument('--sidex', help='grid size = side*2+1', default=2, type=int)
parser.add_argument('--sidey', help='grid size = side*2+1', default=2, type=int)
parser.add_argument('--ipop', help='initial population', default=0, type=int)
parser.add_argument('--irad', help='initial radius', default=10, type=float)
parser.add_argument('--idensity', help='initial density', default=None, type=float)
parser.add_argument('--tol', help='lmax tolerance', default=1, type=float)
parser.add_argument('--rad', help='radius', default=0.5, type=float)
parser.add_argument('--advance', help='# gen to advance on ATH', default=1, type=int)
parser.add_argument('--init', help='initial pattern', default=None, type=str)
parser.add_argument('--seed', help='random seed', default=None, type=int)
parser.add_argument('--memory', help='garbage collection limit in MB', default=60000, type=int)
parser.add_argument('--results', help='results directory', default='./results')
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
if args.seed is None:
    args.seed = random.randint(1,1000000)
random.seed(args.seed)
np.random.seed(args.seed)
print(args)

sess = lifelib.load_rules("b3s23")
lt = sess.lifetree(memory=args.memory)

def log(hdr):
    larr = np.array([x[0] for x in tree])
    lmean = np.mean(larr)
    #lmedian = np.median(leaves)
    #lstd = np.std(leaves)
    top = np.percentile(larr,args.top)
    #print('{:10} wall {} n {:6d} k {:6d} LIFE {:9.0f} ath {:9.0f} pop {:6d} m {:6d} rate {:8.4f} node {:6d} back {:6d} leaves {:6d} depth {:4d} init {:4d} prune {:9.0f} lmax {:9.0f} bt {:>6.2f} uniq {:6d} tleaf {:8.4f} r {:4.0f}'.format(hdr,datetime.datetime.now(),n,k,l,ath,pat.population,m,args.rate,len(tree.all_nodes()),nback,len(tree.leaves()),tree.depth(),len(init),lprune,lamax,bt,len(uniq),args.tleaf,r))
    #print('{:10} wall {} n {:6d} k {:6d} LIFE {:9.0f} ath {:9.0f} pop {:6d} m {:6d} xor {:8.3f} node {:6d} back {:6d} leaves {:6d} depth {:4d} lmean {:9.0f} uniq {:9.0f} top {:9.0f} sib {:4d} irad {:9.0f}'.format(hdr,datetime.datetime.now(),n,kk,l,ath,pat.population,m,args.xorthresh,len(tree.all_nodes()),nback,len(tree.leaves()),tree.depth(),lmean,len(uniq),top,sib,args.irad))
    print('{:10} wall {} n {:6d} k {:6d} LIFE {:9.0f} ath {:9.0f} pop {:6d} m {:6d} xor {:8.3f} node {:6d} back {:6d} rad {:6.3f} effort {:6.3f} lmean {:9.0f} uniq {:9.0f} top {:9.0f} sib {:4d} r {:9.0f}'.format(hdr,datetime.datetime.now(),n,kk,l,ath,pat.population,m,args.xorthresh,len(tree),nback,args.rad,args.effort,lmean,len(uniq),top,sib,r))

# run soup until population is stable, starting at generation lmax
def lifespan(pat,period,lmax):
        pt = np.zeros(period)
        o = int(max(0,lmax-period))
        pat = pat.advance(o) # jump to lmax
        p0 = pat.population
        for j in range(1000000//period):
            for k in range(period):
                pt[k] = pat.population
                pat = pat.advance(1)
            if (pat.population>(2*p0)) and ((pat.population-p0)>10000): # HEURISTIC: 2x pop growth over lmax pop is suspicious
                return -2 # linear growth
            value,counts = np.unique(pt, return_counts=True) # histogram of population trace
            e = scipy.stats.entropy(counts,base=None) # entropy of population distribution 
            if e<0.682689492*np.log(period): # HEURISTIC: threshold
                return max(0,o+j*period)
        return -1 # runaway

def ash(pat,lmax):
    pat = pat.advance(lmax)
    pat = pat.advance(100000) # allow gliders to escape
    pat = pat[-10000:10000, -10000:10000]
    pat = pat.centre()
    return pat

#transforms = ["flip","rot180","identity","transpose","flip_x","flip_y","rot90","rot270","swap_xy","swap_xy_flip","rcw","rccw"]
# 2 phases X 4 rotations
gliders=[]
gliders.append(np.array([[0,-1],[1,-1],[-1,0],[0,0],[1,1]]))
gliders.append(np.array([[-1,-1],[0,0],[1,0],[-1,1],[0,1]]))
gliders.append(np.array([[0,-1],[0,0],[1,0],[-1,1],[1,1]]))
gliders.append(np.array([[-1,-1],[1,-1],[-1,0],[0,0],[0,1]]))
gliders.append(np.array([[-1,-1],[0,-1],[-1,0],[1,0],[-1,1]]))
gliders.append(np.array([[1,-1],[-1,0],[1,0],[0,1],[1,1]]))
gliders.append(np.array([[-1,-1],[0,-1],[1,-1],[1,0],[0,1]]))
gliders.append(np.array([[0,-1],[-1,0],[-1,1],[0,1],[1,1]]))

def render(pat,pre,post):
    for (x0,y0,s) in pre:
        x0 = int(np.around(x0))
        y0 = int(np.around(y0))
        xy = [[x0+x,y0+y] for (x,y) in gliders[s]]
        xy = np.array(xy)
        #print('xy.shape',xy.shape)
        pat[xy]&=0
        #for (x,y) in gliders[s]:
        #    pat[x0+x,y0+y]=0

    for (x0,y0,s) in post:
        x0 = int(np.around(x0))
        y0 = int(np.around(y0))
        xy = [[x0+x,y0+y] for (x,y) in gliders[s]]
        xy = np.array(xy)
        pat[xy]|=1
        #for (x,y) in gliders[s]:
        #    pat[x0+x,y0+y]=1

# initialize root node
states=[x for x in range(8)]
#states.append(None)
#if args.init=='grid2':
#    init0 = [(x,y,random.choice(states)) for x in range(-args.sidex//2,args.sidex//2,args.spacex) for y in range(-args.sidey//2,args.sidey//2,args.spacey)]
#    pat = lt.pattern()
#    render(pat,init0,init0)
#if args.init=='gauss':
#    p=0
#    while p!=args.ipop*5:
#        pat = lt.pattern()
#        init0 = [(np.random.normal(0,args.irad),np.random.normal(0,args.irad),random.choice(states)) for i in range(args.ipop)]
#        render(pat,init0,init0)
#        p=pat.population
#        if not args.legalize:
#            break
#bb = pat.bounding_box
#pat.write_rle('{}/init_seed{:09d}.rle'.format(args.results,args.seed), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)

#pat = lt.pattern()
#for j in range(8):
#    init=[(j*50,0,j)]
#    render(pat,init,init)
#bb = pat.bounding_box
#pat.write_rle('{}/init_seed{:09d}.rle'.format(args.results,args.seed), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
#exit()

def sample():
    #(x0,y0) = random.choice([(-r//2,0),(r//2,0)])
    (x0,y0) = (0,0)
    r=args.irad
    #r = random.choice([100,150])
    #r = 100
    theta = np.random.choice(np.linspace(0,2*np.pi,args.np,endpoint=False))
    x = x0+r*np.cos(theta)
    y = y0+r*np.sin(theta)
    return x,y

samptot=0
prune=0
lmax=0
top=0
ath=0
n=0
k=0
nath=0
maxpop=0
#tree=Tree()
tree=[]
#root=tree.create_node()
#root.tag=0

def orient(x,y):
    if x>0 and y>0:
        #s=random.choice([0,4])
        s=0
    elif x<0 and y<0:
        #s=random.choice([1,5])
        s=1
    elif x<0 and y>0:
        #s=random.choice([2,6])
        s=2
    else:
        #s=random.choice([3,7])
        s=3
    return(s)

#def gen_gcoord(gvec=None):
#    gcoord=[]
#    theta = np.linspace(0,2*np.pi,args.gvec,endpoint=False)
#    if args.gcoord=='single':
#        for i in range(args.gvec):
#            (x0,y0) = (0,0)
#            r=args.irad
#            x = x0+r*np.cos(theta[i])
#            y = y0+r*np.sin(theta[i])
#            gcoord.append((x,y,orient(x,y)))
#    
#    if args.gcoord=='binary':
#        split = int(args.gvec*(args.irad/(args.irad+args.irad2)))
#        inner = np.linspace(0,2*np.pi,split,endpoint=False)
#        outer = np.linspace(0,2*np.pi,args.gvec-split,endpoint=False)
#        for i in range(split):
#            (x0,y0) = (0,0)
#            r=args.irad
#            x = x0+r*np.cos(inner[i])
#            y = y0+r*np.sin(inner[i])
#            gcoord.append((x,y,orient(x,y)))
#        for i in range(args.gvec-split):
#            (x0,y0) = (0,0)
#            r=args.irad2
#            x = x0+r*np.cos(outer[i])
#            y = y0+r*np.sin(outer[i])
#            gcoord.append((x,y,orient(x,y)))
#
#    if args.gcoord=='concentric':
#        #rings = np.random.randint(5,20)
#        #space = np.random.randint(10,20)
#        #rings = np.random.randint(3+0,5+0)
#        #space = np.random.randint(5,10)*2
#        rings = args.gcrings
#        space = args.gcspace
#        perim=2*np.pi*np.sum(np.arange(args.gcinner,args.gcinner+rings*space,space))
#        ng = [np.round((args.gvec/perim)*2*np.pi*(args.gcinner+i*space)) for i in range(rings)]
#        for i in range(rings):
#            theta = np.linspace(0,2*np.pi,int(ng[i]),endpoint=False)
#            for ii in range(int(ng[i])):
#                (x0,y0) = (0,0)
#                r=args.gcinner+i*space
#                x = x0+r*np.cos(theta[ii])
#                y = y0+r*np.sin(theta[ii])
#                gcoord.append((x,y,orient(x,y)))
#
#    if args.gcoord=='multi':
#        perim=2*np.pi*np.sum(np.arange(args.gcinner,args.gcinner+args.gcrings*args.gcspace,args.gcspace))
#        #print('perim',perim)
#        ng = [np.round((gvec/perim)*2*np.pi*(args.gcinner+i*args.gcspace)) for i in range(args.gcrings)]
#        #print('ng',ng)
#        for i in range(args.gcrings):
#            theta = np.linspace(0,2*np.pi,int(ng[i]),endpoint=False)
#            for ii in range(int(ng[i])):
#                (x0,y0) = (0,0)
#                r=args.gcinner+i*args.gcspace
#                x = x0+r*np.cos(theta[ii])
#                y = y0+r*np.sin(theta[ii])
#                gcoord.append((x,y,orient(x,y)))
#        
#    if args.gcoord=='gaussian':
#        for i in range(args.gvec):
#            x = np.random.normal(0,args.irad)
#            y = np.random.normal(0,args.irad)
#            #gcoord.append((x,y,orient(x,y)))
#            gcoord.append((x,y,np.random.randint(8)))
#
#    if args.gcoord=='linear':
#        #lin = np.linspace(-args.irad//2, +args.irad//2, args.gvec)
#        lin = np.linspace(-args.irad//4, +args.irad//4, args.gvec//2)
#        #for i in range(args.gvec):
#        for i in range(args.gvec//2):
#            x = lin[i]
#            y = 0
#            gcoord.append((x,y,None))
#            x = 0
#            y = lin[i]
#            gcoord.append((x,y,None))
#        #print('gen_gcoord',gcoord)
#
#    return gcoord
#
#pat = lt.pattern()
#render(pat,gcoord,gcoord)
#fn = '{}/init_R{:06d}_G{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,int(args.irad),int(args.gvec),args.seed,n)
#bb = pat.bounding_box
#pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)

#def retry(l):
#    #return 0.5*np.sqrt(l)
#    return np.log(l)

j=0
uniq={}
while j<args.iroot:
    #print('iroot',j)
    #init = [(np.random.normal(0,args.irad),np.random.normal(0,args.irad),random.choice(states)) for i in range(args.ipop)]
    #init = [0]*args.gvec
    #for i in range(args.ipop):
    #    init[np.random.randint(args.gvec)]=1
#    if args.gcoord=='multi':
#        gcoord = gen_gcoord(j+100)
#    else:
#        gcoord = gen_gcoord()
#    #print('gvec',args.gvec,'gcoord',len(gcoord))
#    if args.mut=='ones':
#        init = [1]*len(gcoord)
#    if args.mut=='xor':
#        init = np.random.uniform(size=len(gcoord))
#        init = np.where(init < args.xorthresh, 1, 0)
#
#    if args.gcoord=='linear':
#        #init = np.random.uniform(size=len(gcoord))
#        #init = np.where(init < args.xorthresh, 1, 0)*4 + np.random.randint(4)
#        #init = np.random.randint(8*2,size=len(gcoord))
#        init = np.random.randint(8,size=len(gcoord))
#        
    pat = lt.pattern()
    #print('gcoord',len(gcoord))
    #print('init',init)
    #initgc = [gcoord[i] for i in range(args.gvec) if init[i]]
    #initgc = [(gcoord[i][0],gcoord[i][1],init[i]) for i in range(len(gcoord)) if init[i]<8]
    #print('initgc',initgc)
    initgc=[]
    #initgc.append((0,0,np.random.randint(8)))
    #for i in range(args.ipop):
    i=0
    while i<args.ipop:
        x = np.random.normal(0,args.irad2)
        y = np.random.normal(0,args.irad2)
        overlap=False
        for (x0,y0,_) in initgc:
            if abs(x-x0)<4 or abs(y-y0)<4: # prevent overlaps
                #print('x',x,'y',y,'x0',x0,'y0',y0)
                overlap=True
        if not overlap:
            initgc.append((x,y,orient(x,y)))
            i+=1
    #initgc = [(np.random.normal(0,args.irad),np.random.normal(0,args.irad),np.random.randint(8)) for i in range(args.ipop)]
    print('initgc',initgc)
    render(pat,initgc,initgc)
    #p1 = pat.population
    #if args.mut=='xor':
    #    init = np.random.uniform(size=len(gcoord))
    #    init = np.where(init < args.xorthresh, 1, 0)
    #    initgc2 = [(gcoord[i][0],gcoord[i][1],gcoord[i][2]+init[i]*4) for i in range(len(gcoord))]
    #    render(pat,initgc,initgc2)
    #    p2 = pat.population

    #if args.gcoord=='gaussian' and ((p1 != 5*args.gvec) or (p2 != 5*args.gvec)):
    #    print('retry','p1',p1,'p2',p2)
    #    continue
    l = lifespan(pat,800,0)
    if l>0:
        print('init',np.mean(np.array(initgc)),'l',l)
    if l>args.mininit:
        d = pat.digest()
        if d not in uniq:
            uniq[d]=True
            #node=tree.create_node(tag=l, identifier=d, parent=root, data=[initgc,1]) # gliders, depth
            tree.append((l,initgc))
            print('INIT','j',j,len(tree),'l',l,'d',d,'p',pat.population)
            if args.keep or j==0:
                fn = '{}/init_P{:06d}_L{:06d}_d{:d}_n{:09d}.rle'.format(args.results,pat.population,int(l),d,n)
                bb = pat.bounding_box
                pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
            j+=1

#pool = [x for x in tree.all_nodes() if x!=root]
#larr = np.array([float(i.tag) for i in pool]) # lifespan distribution
#if args.alt2:
#    larr = np.multiply(larr, np.log(larr))
#if args.alt3:
#    larr = np.square(larr)
#larr = np.power(larr,args.pow)
#prob = larr / np.sum(larr)
#root.data = init0
#node=root
#node=random.choice(tree.leaves())
depth=0
sib=0
nback=0
r = args.irad
nroot=0
lprune=100
lmean=0
lmedian=0
lstd=0
lamax=0
bot=0
#ref=node.data.copy()
m=1
bt=args.backtrack
#nsample=0
ns_prune=0
nsmean=0
kk=0
sib=0
while True:
    nath+=1
    k=0

#    if args.prune:
#        pool = [x for x in tree.all_nodes() if x!=root]
#        ns = [node.data[1] for node in pool]
#        nsmean = np.mean(ns)
#        leaves = np.array([float(i.tag) for i in tree.leaves()]) # lifespan distribution
#        lmean = np.mean(leaves)
#        larr = np.array([int(i.tag) for i in pool]) # lifespan distribution
#        if args.alt2:
#            larr = np.multiply(larr, np.log(larr))
#        if args.alt3:
#            larr = np.multiply(larr, np.sqrt(larr))
#        if args.alt4:
#            larr = np.multiply(larr, larr)
#        if args.alt5:
#            larr = np.power(larr, args.exp)
#        for node in pool:
#            #if node.is_leaf() and node.data[1]<=0 and float(node.tag)<lmean:
#            #if ns_prune>retry(float(node.tag)):
#            #if nsmean>args.nsprune and ns_prune>nsmean*args.backtrack:
#            #if nsmean>args.nsprune and ns_prune>nsmean*args.backtrack and float(node.tag)<lmean:
#            l = float(node.tag)
#            ns_prune = node.data[1]
#            #if ns_prune>args.nsprune and l<lmean:
#            #if ns_prune>args.nsprune:
#            if ns_prune>np.log(l):
#                if node.is_leaf():
#                    tree.remove_node(node.identifier)
#                    log('PRUNE')
#                    if tree.size()>2:
#                        continue
#                    else:
#                        exit()
#                else:
#                    continue

    #pool = [x for x in tree.all_nodes() if x!=root]
    #larr = np.array([int(i.tag) for i in pool]) # lifespan distribution

    if len(tree)==0:
        break
    larr = np.array([x[0] for x in tree]) # lifespan distribution
    imax = np.max(larr)
    marr = [i for i,x in enumerate(tree) if x[0]==imax]
    sib = len(marr)

    #if sib>1 and random.expovariate(1)>args.rate:
    #for bb in range(int(random.expovariate(1))):
#    for bb in range(np.log(sib)):
#        if len(marr)>0:
#            ii = np.random.randint(len(marr))
#            del tree[marr[ii]]
#            del marr[ii]
#            #larr = np.array([x[0] for x in tree]) # lifespan distribution
#            #imax = np.max(larr)
#            #marr = [i for i,x in enumerate(tree) if x[0]==imax]
#            sib = len(marr)
#            log("BLAST")
    #if sib>args.sib:
    #    while len(marr)>0:
    #        del(tree[marr[0]])
    #        marr = [i for i,x in enumerate(tree) if x[0]==imax]
    #    log("BLAST")
    #    continue
    #node = random.choice(marr)

    #nodes = sorted(pool, reverse=True, key=lambda node: (int(node.tag),node.data[1]))
    #node = nodes[0]
    if len(tree)==0:
        break
    myself=len(tree)-1
    node=tree[myself]

    #nodes = sorted(pool, reverse=True, key=lambda node: int(node.tag))
    #node = nodes[0]
#    ns = [node.data[1] for node in pool]
#    nsmean = np.mean(ns)
#    leaves = np.array([float(i.tag) for i in tree.leaves()]) # lifespan distribution
#    lmean = np.mean(leaves)
#    larr = np.array([int(i.tag) for i in pool]) # lifespan distribution
#    if args.alt2:
#        larr = np.multiply(larr, np.log(larr))
#    if args.alt3:
#        larr = np.multiply(larr, np.sqrt(larr))
#    if args.alt4:
#        larr = np.multiply(larr, larr)
#    if args.alt5:
#        larr = np.power(larr, args.exp)
    #nodes = sorted(pool, reverse=True, key=lambda node: int(node.tag))

    #nodes = sorted(pool, reverse=True, key=lambda node: (int(node.tag),-node.data[2]))

    #nodes = list(tree.expand_tree(root.identifier,Tree.WIDTH))
    #nodes = [tree.get_node(nid) for nid in tree.expand_tree(root.identifier,Tree.WIDTH,reverse=True)]
    #nodes = [tree.get_node(nid) for nid in tree.expand_tree(root.identifier,Tree.ZIGZAG,reverse=True)]
    #nodes = [tree.get_node(nid) for nid in tree.expand_tree(root.identifier,Tree.DEPTH,reverse=True)]

    #ns_prune = np.percentile(ns,args.ns_perc)
    #leaves = np.array([float(i.tag) for i in tree.leaves() if i.data[1]>0]) # lifespan distribution
#    prob = larr / np.sum(larr)
#    node = np.random.choice(pool,p=prob)
    #imax = min(len(nodes),int(np.floor(random.expovariate(1))))
    #node = nodes[imax]
    #if node==root:
    #    continue
    nback +=1
    #l=int(node.tag)
    l=node[0]
    lmax=l
    #bt=np.log(l)
    #ref = node.data[0]
    ref = node[1]
    #gcoord = node.data[3]
    #gvec = len(gcoord)
    #refgc = [gcoord[i] for i in range(args.gvec) if ref[i]]
    #refgc = [(gcoord[i][0],gcoord[i][1],gcoord[i][2]+ref[i]*4) for i in range(gvec)]
    #refgc = [(gcoord[i][0],gcoord[i][1],ref[i]) for i in range(gvec) if ref[i]<8]
    #ns_prune = node.data[1]
    #nsample = node.data[1]
    #nsample+=1
    pat=lt.pattern()
    render(pat,ref,ref)
    #if args.debug:
    #    tree.show()
    log('BACK')
    #for kk in range(int(1*np.log(float(node.tag)))):
    #for kk in range(int(args.effort)):
    #for kk in range(int(args.effort*np.log(float(node.tag)))):
    #for kk in range(int(args.effort*np.log(node[0]))):
    for kk in range(int(args.backtrack)):
        # mutate the current node
        mut = ref.copy()
        if args.multimut:
            m = int(np.ceil(random.expovariate(1)))
        else:
            m=1
        #for i in random.choices(range(len(mut)), k=m):
        for i in range(m):
            #p0 = np.random.uniform()
            #p0=1
            #if p0<0.5:
            if True:
                #r = args.irad
                #r = 2*np.log(lmax)
                #r = args.rad*np.sqrt(lmax)
                r = np.power(lmax,args.rad)
            else:
                r = args.irad2

            p = np.random.uniform()
            if p<args.xorthresh:
                if len(mut)>0:
                    ii = np.random.randint(len(mut))
                    del mut[ii]
            else:
                #mut.append((np.random.normal(0,args.irad),np.random.normal(0,args.irad),np.random.randint(8)))
                #r = np.sqrt(lmax)*args.rate
                #r = args.irad
                x = np.random.normal(0,r)
                y = np.random.normal(0,r)
                #if args.orient:
                #if p0<0.5:
                if True:
                    mut.append((x,y,orient(x,y)))
                else:
                    mut.append((x,y,np.random.randint(8)))
#            gcoord.append((x,y,orient(x,y)))
#            if args.gcoord=='gaussian':
#                ii = np.random.randint(gvec)
#                mut[ii]=9 # clear
#                #mut[ii] = np.random.randint(100)
#                #if mut[ii] < 8:
#                #    mut[ii] = 15
#                #else:
#                #    #mut[ii] = np.random.randint(16)
#                #    mut[ii] = np.random.randint(8)
#            elif args.gcoord=='linear':
#                ii = np.random.randint(gvec)
#                #if mut[ii] < 8:
#                if False:
#                    mut[ii] = 15
#                else:
#                    #mut[ii] = np.random.randint(16)
#                    mut[ii] = np.random.randint(8)
#            elif args.mut=='xor':
#                mut[np.random.randint(gvec)]^=1
#            elif args.mut=='ones':
#                ones = [j for j in range(gvec) if mut[j]==1]
#                if len(ones)>0:
#                    mut[random.choice(ones)]=0
        #mutgc = [gcoord[i] for i in range(args.gvec) if mut[i]]
        #mutgc = [(gcoord[i][0],gcoord[i][1],gcoord[i][2]+mut[i]*4) for i in range(gvec)]
        #mutgc = [(gcoord[i][0],gcoord[i][1],mut[i]) for i in range(gvec) if mut[i]<8]
    
        # create new pattern with glider config specified by mut
        render(pat,ref,mut)
        l = lifespan(pat,100,lmax*args.tol)
        n+=1
    
        blast=False
        if l==-1:
            log('RUNAWAY')
            bb = pat.bounding_box
            pat.write_rle('{}/runaway_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
        elif l==-2:
            log('GROWTH')
            bb = pat.bounding_box
            pat.write_rle('{}/growth_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
            break
        elif l==lmax:
            ref=mut
            # add harmless mut as sibling
            d = pat.digest()
            if d not in uniq:
                uniq[d]=True
                #node.data[1]=0
                #tree.show(idhidden=True)
                #print('node',node)
                #print('parent',tree.parent(node))
                #node = tree.create_node(tag=l, identifier=d, parent=node, data=[ref])
                #tree.create_node(tag=l, identifier=d, parent=node, data=[ref,node.data[1]+1])
                tree.append((l,ref))
                # statistics
                #pool = [x for x in tree.all_nodes() if x!=root]
                #larr = np.array([float(i.tag) for i in pool]) # lifespan distribution
                #sarr = np.array([float(i.data[1]) for i in pool]) # nsample distribution
                #samptot = np.mean(sarr)
                #larr = np.log(larr)
                #prob = larr / np.sum(larr)
                #leaves = np.array([float(i.tag) for i in tree.leaves()]) # lifespan distribution
                #lmean = np.mean(leaves)
                #lmedian = np.median(leaves)
                #lstd = np.std(leaves)
                #top = np.percentile(leaves,args.top)
                #tree = [x for x in tree if ((x[0]<imax) and (np.random.uniform()<0.5))]
                log('BEST')
                #break
                #k=0
                #prune=False
            else:
                #node.data=[node.data[0],node.data[1]+1,node.data[2],node.data[3]]
                log('DUP1')
                #prune=True
            #node.data=[mut,node.data[1],pat.population,node.data[3]] # keep harmless mutations
            #break
            sib+=1
            if sib>r:
                blast=True
        
            #if sib>args.blast*np.sqrt(imax):
        elif l>lmax:
            ref=mut
            # add new lmax to tree
            d = pat.digest()
            if d not in uniq:
                uniq[d]=True
                #node.data = [node.data[0],0,node.data[2],node.data[3]]
                #node = tree.create_node(tag=l, identifier=d, parent=node, data=[ref])
                #tree.create_node(tag=l, identifier=d, parent=node, data=[ref,node.data[1]+1])
                tree.append((l,ref))
                sib=0
                # statistics
                #pool = [x for x in tree.all_nodes() if x!=root]
                #larr = np.array([float(i.tag) for i in pool]) # lifespan distribution
                #sarr = np.array([float(i.data[1]) for i in pool]) # nsample distribution
#samptot = np.mean(sarr)
#larr = np.log(larr)
                #prob = larr / np.sum(larr)
                #leaves = np.array([float(i.tag) for i in tree.leaves()]) # lifespan distribution
                #lmean = np.mean(leaves)
                #lmedian = np.median(leaves)
                #lstd = np.std(leaves)
                #top = np.percentile(leaves,args.top)
                if l>ath:
                    ath = l
                    fn = '{}/ATH_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,pat.population,int(l),args.seed,n)
                    bb = pat.bounding_box
                    pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
                    log('ATH')
                    #tree.show(idhidden=True)
                    print([x[0] for x in tree])
                    nath=0
                #elif l==lmax:
                #    log('BEST')
                else:
                    if l>top:
                        fn = '{}/lmax_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,pat.population,int(l),args.seed,n)
                        bb = pat.bounding_box
                        pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
                    log('LMAX')
                    #tree.show(idhidden=True)
                #break
            else:
                #node.data=[node.data[0],node.data[1]+1,node.data[2],node.data[3]]
                log('DUP2')
                #prune=True
                #print('prune',prune,node.is_leaf())
                #break

            lmax=l
            k=0
            #prune=False
            #break
            #if sib>args.blast*np.sqrt(imax):

        else:
            #log('REVERT')
            render(pat,mut,ref) # revert
            #node.data=[node.data[0],node.data[1]+1,node.data[2],node.data[3]]
        #k+=1
        #prune=True

        if blast:
            larr = np.array([x[0] for x in tree]) # lifespan distribution
            imax = np.max(larr)
            marr = [i for i,x in enumerate(tree) if x[0]==imax]
            #tree = [x for x in tree if x[0]<imax and np.random.uniform()<0.5]
            #del tree[-1*max(1,np.random.randint(len(marr))):]
            #del tree[-1*(1+np.random.randint(len(marr)//2,len(marr))):]
            del tree[-1*len(marr):]
            marr = [i for i,x in enumerate(tree) if x[0]==imax]
            sib = len(marr)
            log("BLAST")
            break

    # PRUNE
#    log('PRUNE')
#    del(tree[myself])
#
#    if len(tree)==0:
#        break
    
#    larr = np.array([x[0] for x in tree]) # lifespan distribution
#    imax = np.max(larr)
#    marr = [i for i,x in enumerate(tree) if x[0]==imax]
#    sib = len(marr)
#
#    #if sib>args.blast*np.sqrt(imax):
#    if sib>r:
#        #tree = [x for x in tree if ((x[0]<imax) and (np.random.uniform()<0.5))]
#        tree = [x for x in tree if x[0]<imax]
#        larr = np.array([x[0] for x in tree]) # lifespan distribution
#        if len(larr)==0:
#            break
#        imax = np.max(larr)
#        marr = [i for i,x in enumerate(tree) if x[0]==imax]
#        sib = len(marr)
#        log("BLAST")
#        print([x[0] for x in tree])

    #for bb in range(int(np.log(sib))):
    #    if len(marr)>0:
    #        ii = random.choice(marr)
    #        del tree[ii]
    #        larr = np.array([x[0] for x in tree]) # lifespan distribution
    #        imax = np.max(larr)
    #        marr = [i for i,x in enumerate(tree) if x[0]==imax]
    #        sib = len(marr)
    #        log("BLAST")
    #print('prune',prune,node.is_leaf())
    #if prune and node.is_leaf():
    #tree.show(idhidden=True)
    #print('node',node)
    #for nn in tree.leaves(node):
    #if node.is_leaf():
#    prune_yourself = node.is_leaf()
#    if prune_yourself:
#        #print('nn',nn)
#        #d = pat.digest()
#        #if d in uniq:
#        #    del uniq[d]
#        #    #uniq.pop(d, None)
#        #tree.remove_node(node.identifier)
#        if node==root:
#            continue
#        tree.remove_node(node.identifier)
#        log('PRUNE')
#        if tree.size()>2:
#            continue
#        else:
#            print('ERROR 1')
#            exit()
