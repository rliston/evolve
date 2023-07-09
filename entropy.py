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
parser.add_argument('--rootlife', help='lifespan of root', default=0, type=int)
parser.add_argument('--alt3', default=False, action='store_true')
parser.add_argument('--alt2', default=False, action='store_true')
parser.add_argument('--alt', default=False, action='store_true')
parser.add_argument('--backtrack', help='backtrack threshold', default=1, type=float)
parser.add_argument('--btboost', help='backtrack increase every lmax', default=1, type=float)
parser.add_argument('--binary', default=False, action='store_true')
parser.add_argument('--minleaves', help='minimum pool size, else root', default=22, type=int)
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
parser.add_argument('--ipop', help='initial population', default=10, type=int)
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

def log(hdr,n,k,l,ath,pop,m,r,nath):
#    d=0
#    if len(digest)>0:
#        d=max(digest.values())
    #print('all_nodes',len(tree.all_nodes()))
    #if width is not None and len(width)>args.topk:
#    if len(pool)>0:
#        #h = [int(tree[i].tag) for i in pool[1:topk]]
#        h = [int(i.tag) for i in pool[0:topk]]
#    else:
#        h=[0]

    #print('topk',topk,'pool',len(pool), pool[0:topk])
#    if topk==0:
#        h=[0]
#    else:
#        h = [int(i.tag) for i in pool[0:topk]]
    #print('h',h)

#    if len(tree.all_nodes()) > 5:
#        h = [tree[i].data[1] for i in list(tree.expand_tree(mode=Tree.WIDTH))[-5:]]
#    else:
#        h=''
    print('{:10} wall {} n {:6d} k {:6d} LIFE {:12.4f} ath {:12.4f} pop {:6d} m {:6d} r {:12.8f} nath {:6d} node {:6d} back {:6d} leaves {:4d} depth {:4d} backtrack {:12.4f} init {:4d} lmean {:12.8f} pmean {:12.8f}'.format(hdr,datetime.datetime.now(),n,k,l,ath,pop,m,r,nath,len(tree.all_nodes()),nback,len(tree.leaves()),tree.depth(),bt,len(init),np.mean(lmean),np.mean(pmean)))
    #print('{:10} wall {} n {:6d} k {:6d} LIFE {:12.4f} ath {:12.4f} pop {:6d} m {:6d} r {:12.8f} nath {:6d} node {:6d} back {:6d} density {:8.4f} leaves {:4d} depth {:4d} backtrack {:3d} init {:4d} pool {:6d}'.format(hdr,datetime.datetime.now(),n,k,l,ath,pop,m,r,nath,len(tree.all_nodes()),nback,pop/(args.sidex*args.sidey),len(tree.leaves()),tree.depth(),args.backtrack,len(init),len(pool)))
    #print('{:10} wall {} n {:6d} k {:6d} LIFE {:12.4f} ath {:12.4f} pop {:6d} m {:6d} r {:12.8f} nath {:6d} node {:6d} back {:6d} density {:8.4f} leaves {:4d} depth {:4d} backtrack {:3d} init {:4d} lavg {:12.4f} lstd {:12.4f}'.format(hdr,datetime.datetime.now(),n,k,l,ath,pop,m,r,nath,len(tree.all_nodes()),nback,pop/(args.sidex*args.sidey),len(tree.leaves()),tree.depth(),args.backtrack,len(init),lavg,lstd))
#    print('{:10} wall {} n {:6d} k {:6d} LIFE {:12.4f} ath {:12.4f} pop {:6d} m {:6d} r {:12.8f} nath {:6d} node {:6d} back {:6d} density {:8.4f} leaves {:4d} depth {:4d} backtrack {:3d} init {:4d} topk {:4d} {:8.4f} min {} max {}'.format(hdr,datetime.datetime.now(),n,k,l,ath,pop,m,r,nath,len(tree.all_nodes()),nback,pop/(args.sidex*args.sidey),len(tree.leaves()),tree.depth(),args.backtrack,len(init),topk,args.topk,min(h),max(h)))
#print('{:10} wall {} n {:6d} k {:6d} LIFE {:12.4f} ath {:12.4f} pop {:6d} m {:6d} r {:12.8f} nath {:6d} nbirth {:6d} nback {:6d} density {:12.8f} topk {:6d} leaves {:6d} backtrack {} init {} {}'.format(hdr,datetime.datetime.now(),n,k,l,ath,pop,m,r,nath,len(history),nback,pop/(args.sidex*args.sidey),topk,len(tree.leaves()),args.backtrack,len(init),[x[0] for x in history[0:5]]))

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

sess = lifelib.load_rules("b3s23")
lt = sess.lifetree(memory=args.memory)
pat = lt.pattern()

if args.init is not None and args.init[-4:] == ".rle":
    pat = lt.load(args.init) # empty pattern if None, else load .rle file
    pat = pat.centre()
    init0 = pat.coords()
else:
    pat = lt.pattern()
    # initial conditions
    if args.init=="normal":
        if args.idensity is None:
            init0 = np.random.normal(0,args.irad,size=[args.ipop,2])
        else:
            init0 = np.random.normal(0,args.irad,size=[int((args.idensity*np.pi*args.irad*args.irad)/args.rad),2])
    elif args.init=="uniform":
        init0 = np.random.uniform(low=-args.irad, high=args.irad,size=[int(args.idensity*args.irad*2*args.irad*2),2])
    elif args.init=="grid":
        init0 = np.array([(x,y) for x in range(-args.side*args.space,args.side*args.space+args.space,args.space) for y in range(-args.side*args.space,args.side*args.space+args.space,args.space)])
    elif args.init=="grid2":
        init0 = np.array([(x,y) for x in range(-args.sidex//2,args.sidex//2,args.spacex) for y in range(-args.sidey//2,args.sidey//2,args.spacey)])
    elif args.init=="grid3":
#        init0 = [(x,y) for x in range(-args.sidex//2,args.sidex//2,args.spacex) for y in range(-args.sidey//2,args.sidey//2,args.spacey)]
        init0 = []
        for x in range(-args.sidex//2,args.sidex//2,args.spacex):
            for y in range(-args.sidey//2,args.sidey//2,args.spacey):
                if x<-args.keepout or x>=args.keepout or y<-args.keepout or y>=args.keepout:
                    init0.append((x,y))
#        for (x,y) in init0:
#            if abs(x)>args.keepout or abs(y)>args.keepout:
#                init.append((x,y))
#        if args.orad is not None:
#            init.append((-args.spacex//2,-args.spacey//2))
        init0 = np.array(init)
    else:
        init0=np.array([[0,0]])
    #pat[init] |=1
    #fn = '{}/init_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,pat.population,0,args.seed,0)
    #bb = pat.bounding_box
    #pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)

init0 = init0.astype(float)
init = np.copy(init0)
#lmax=args.lmax
lmax=0
ath=0
n=0
k=0
nath=0
maxpop=0
#width=None
#history=[]
pool=[]
tree=Tree()
root=tree.create_node()
root.tag = args.rootlife
#root.data=(init0,0,args.rad,init0)
root.data=(init0,0,args.backtrack,np.copy(init0))
node=root
topk=0
nback=0
lmean=[]
pmean=[]
#digest={}
#lavg=0
#lstd=0
#r = args.rad
r = args.rad
bt = args.backtrack
while True:
    n+=1
    nath+=1
    k+=1
    #r = node.data[2]

    #if args.showtree and ((n%1000)==0):
    #    tree.show(idhidden=True)
    #    #tree.show(idhidden=False)

#    d = pat.digest()
#    if d not in digest:
#        digest[d]=1
#    else:
#        digest[d]+=1

    # random xor mutation
    #coords = pat.coords()
#    if len(coords)==0: # ensure at least 1 live cell
#        coords = [[0,0]]

#    if lmax>0:
#        args.backtrack = np.sqrt(lmax)*args.effort

    #if k>args.backtrack and len(tree.all_nodes())>1:
    #if (n%args.backtrack)==0 and len(tree.all_nodes())>1:
        #lmax=0
    if k>bt:
        k=0
        nback +=1
        if pat.population > maxpop:
            log('MAXPOP',n,k,l,ath,pat.population,m,r,nath)
            bb = pat.bounding_box
            pat.write_rle('{}/maxpop_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
            maxpop = pat.population
        #bt = args.backtrack
#        args.side +=1
#        init = np.array([(x,y) for x in range(-args.side*args.space,args.side*args.space+args.space,args.space) for y in range(-args.side*args.space,args.side*args.space+args.space,args.space)])

#        if len(history)<=1:
#            print("BACKTRACK STACK EMPTY")
#            break
        #print([x[1] for x in history[0:topk]])
        #topk = int(len(history)*args.topk)
        #topk = max(args.backtrack, topk)
        #topk = min(len(history), topk)
        #(xy,lmax) = history[np.random.randint(topk)]

        #topk = int(np.around(len(history)*(1.-args.topk)))
        #topk = int(random.expovariate(1))
        #topk = 1.0/len(history)
        #topk = 1.0/np.around(len(history)*(1.-args.topk))
        #topk = int(random.expovariate(topk))
        #topk = min(len(history)-1, topk)
        #(xy,lmax) = history[topk]
        #topk = int(np.ceil(len(history)*(1.-args.topk)))
        #print(','.join([tree[node].tag for node in tree.expand_tree(mode=Tree.DEPTH)]))

        # for dev in reversed(list(self._tree.expand_tree(mode=Tree.WIDTH))[1:]):
        #node = tree[random.choice(list(tree.expand_tree(mode=Tree.WIDTH))[0:5])]
        #(xy,lmax) = node.data
        #print('node',node,'width',list(tree.expand_tree(mode=Tree.WIDTH))[0:5])
        #print('node',node,'width',[v.data[1] for v in list(tree.expand_tree(mode=Tree.WIDTH))[0:5]])
        #print('node',node,'width',[tree[v].data[1] for v in list(tree.expand_tree(mode=Tree.WIDTH))[0:5]])
        #print('width',len(list(tree.expand_tree(mode=Tree.WIDTH))), [tree[i].data[1] for i in list(tree.expand_tree(mode=Tree.WIDTH))[-5:]])

        #pool = tree.leaves()
        #pool = [tree[i] for i in tree.expand_tree(mode=Tree.WIDTH)]
        #pool.pop(0) # remove root
        #print('pool[-1]',type(pool[-1]),dir(pool[-1]),'len',len(pool))
        #print([node.tag for node in pool])
        #pool = sorted(pool, reverse=True, key=attrgetter('tag'))
        #print([node.tag for node in pool])
#        pool = tree.all_nodes()
#        pool = sorted(pool, reverse=True, key=lambda node:int(node.tag))

        #width = list(tree.expand_tree(mode=Tree.WIDTH))
        #width = list(tree.expand_tree(mode=Tree.WIDTH))
        #if len(width) > 1:
        #    width.pop(0)
        #topk = min(args.topk,len(width))
        #topk = int(len(width)*args.topk)
        #topk = int(np.sqrt(len(tree.leaves())-1))
        #topk = int(np.sqrt(len(pool))/args.topk)
        #topk = int(len(pool)*args.topk)
        #topk = int(np.power(len(pool), args.topk))
#        topk = int(len(pool)/np.log(len(pool)))
        #node = random.choice([tree[i] for i in list(tree.expand_tree(mode=Tree.WIDTH))[-topk:]]) # width, sort ties by digest
        #node = random.choice([tree[i] for i in pool[0:topk]]) # leaves, reverse sort by lmax
#        node = random.choice(pool[0:topk]) # leaves, reverse sort by lmax
#        if node != root:
#            (xy,lmax) = node.data
#        if np.random.rand()<0.1:
#            node = random.choice(tree.all_nodes())
#        else:
#            node = random.choice(tree.leaves())
#        if tree.depth()>1 and random.choice([True,False]):
#            node = tree.parent(node.identifier)
#        (xy,lmax) = node.data
#        if len(tree.leaves()) > 5:
#            node = random.choice(tree.leaves())
#            if random.choice([True,False]):
#                node = tree[node.predecessor]
#            (xy,lmax) = node.data
#        else:
#            topk = min(len(history), int(args.topk))
#            (lmax,xy,node) = random.choice(history[0:topk])
#        if pat.population > len(node.data[0]):
#            node.data=[pat.coords(),lmax] # retain harmless mutations if they grow
        #if random.expovariate(1) > args.boom:


        #if args.keep:
        #    node.data=(pat.coords(),lmax,r,init) # keep harmless mutations

        pool = tree.all_nodes()
        #topk = min(lmax,np.std(np.array([int(i.tag) for i in pool]))*args.topk)
        #topk = np.std(np.array([int(i.tag) for i in pool]))*args.topk
        #pool = [i for i in pool if int(i.tag)>=topk]
        pool = [i for i in pool if int(i.tag)>0]
        larr = np.array([int(i.tag) for i in pool])
        parr = np.array([len(i.data[0]) for i in pool])+1

        #larr = np.array([int(i.tag) if int(i.tag)>topk for i in pool])
        #pool = sorted(pool, reverse=True, key=lambda node:int(node.tag))
        #topk = int(len(pool)*args.topk)
        #topk = int(np.std(larr)*args.topk)
        #larr = larr[0:topk]
        if args.alt3:
            larr = np.abs(np.multiply(larr/parr,np.log(larr/parr)))
        if args.alt2:
            larr = np.square(larr/parr)
        if args.alt:
            larr = larr/parr
        if args.square:
            larr = np.square(larr)
        if args.log:
            larr = np.multiply(larr,np.log(larr))
        #lavg = np.sqrt(np.mean(larr))
        #lstd = np.sqrt(np.std(larr))
        #prob = larr / sum(larr)
        prob = larr / np.linalg.norm(larr, ord=1)
        #prob = np.linalg.norm(larr,ord=2,keepdims=True)
        #if len(pool)==0 or args.boom:
        #if len(tree.all_nodes())<args.minpool:
        #if len(tree.leaves()) < args.minleaves:
        #if len(tree.all_nodes()) < nback:
        if (nback%100)==0:
            node=root
        else:
            node = np.random.choice(pool,p=prob)

        (xy,lmax,bt,init) = node.data
        #init = np.copy(init0)
        pat = lt.pattern()
        pat[xy] |=1
        log('BACKTRACK',n,k,lmax,ath,pat.population,len(xy),r,nath)
        if args.debug:
            tree.show()
        
    else:
        #m0 = int(np.ceil(random.expovariate(1)))
        #loc = random.choices(coords,k=m)
        #xy = np.array([[np.random.normal(x,r),np.random.normal(y,r)] for i in range(m0) for (x,y) in loc])
        #xy = np.random.uniform(low=-args.side*args.space, high=args.side*args.space+args.space,size=[int(np.ceil(random.expovariate(1))),2])
        #(x,y) = random.choice(coords)
        #(x,y) = init[np.random.randint(len(init))]
        #xy = np.array([[np.random.normal(x,r),np.random.normal(y,r)] for i in range(m)])
        #xy = np.array([[x+((np.abs(np.random.normal(0,r))+1)*random.choice([-1,1])), y+((np.abs(np.random.normal(0,r))+1)*random.choice([-1,1]))] for i in range(m)])

        #m = int(np.ceil(random.expovariate(1)))
        #(x,y) = random.choice(init)
        #xy = np.array([[np.around(np.random.normal(x,r)),np.around(np.random.normal(y,r))] for i in range(m)])

#        xy=[]
#        #for i in range(int(np.ceil(random.expovariate(1)))):
#        #if args.binary:
#        #    for (x,y) in init:
#        #        xy.extend([[np.around(np.random.normal(x,r)),np.around(np.random.normal(y,r))] for i in range(int(np.ceil(random.expovariate(1))))])
#        #else:
#        i = random.choice(range(len(init)))
#        init[i] += np.random.normal(0,args.orad,size=2)
#        (x,y) = init[i]
#        xy.extend([[np.around(np.random.normal(x,r)),np.around(np.random.normal(y,r))] for i in range(int(np.ceil(random.expovariate(1))))])
#
#        #if args.orad is not None:
#        #    xy.extend([[np.around(np.random.normal(-args.spacex//2,args.orad)),np.around(np.random.normal(-args.spacey//2,args.orad))] for i in range(int(np.ceil(random.expovariate(1))))])
#
#        xy = np.array(xy)
#        m = len(xy)

        i = random.choice(range(len(init)))
        init[i] += np.random.normal(0,r,size=2)
        xy = np.around(init[i]).reshape([1,2])
        #xy = np.around(init[i] + np.random.normal(0,r,size=2)).reshape([1,2])
        #print('xy',xy,pat[xy])

        m = 1


    # NEW
#    m1 = np.ceil(random.expovariate(1))
#    if m1 > 6:
#        (x,y) = random.choice(init)
#        pat[x-args.space//2,y-args.space//2:x+args.space//2,y+args.space//2] = 0
#        lmax=0

    v0 = pat[xy]
    pat[xy] ^=1 # apply mutation xy
#    pat[init] |=1 # do not modify initial pattern
    
    l = lifespan(pat,100,lmax*args.tol)

    if l==-1:
        log('RUNAWAY',n,k,l,ath,pat.population,m,r,nath)
        bb = pat.bounding_box
        pat.write_rle('{}/runaway_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
        pat[xy] = v0 # revert mutation xy
        bb = pat.bounding_box
        pat.write_rle('{}/snapshot_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
    elif l==-2:
        log('GROWTH',n,k,l,ath,pat.population,m,r,nath)
        bb = pat.bounding_box
        pat.write_rle('{}/growth_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
        pat[xy] = v0 # revert mutation xy
        bb = pat.bounding_box
        pat.write_rle('{}/snapshot_L{:09d}_seed{:09d}_n{:09d}.rle'.format(args.results,l,args.seed,n), header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
        # add snapshots to tree
        #bt=np.log(l)
#        d = pat.digest()
#        if d not in [v.identifier for v in tree.all_nodes()]: # no dups
#            #node = tree.create_node(tag=l, identifier=d, parent=node, data=[pat.coords(),l,r*args.growth,init])
#            #node = tree.create_node(tag=l, identifier=d, parent=node, data=[pat.coords(),l,r*args.growth,init0])
#            node = tree.create_node(tag=l, identifier=d, parent=node, data=[pat.coords(),l,bt,np.copy(init)])
#            lmean.append(l)
#            pmean.append(pat.population)
    elif l==lmax:
        log('BEST',n,k,l,ath,pat.population,m,r,nath)
        k=0
    elif l>lmax*args.tol:
        # add new lmax to tree
        #bt=np.log(l)
        d = pat.digest()
        #print('DIGEST',d,pat.population,pat.coords())
        if d not in [v.identifier for v in tree.all_nodes()]: # no dups
            #node = tree.create_node(tag=l, identifier=d, parent=node, data=[pat.coords(),l,r*args.growth,init])
            #node = tree.create_node(tag=l, identifier=d, parent=node, data=[pat.coords(),l,r*args.growth,init0])
            node = tree.create_node(tag=l, identifier=d, parent=node, data=[pat.coords(),l,bt,np.copy(init)])
            lmean.append(l)
            pmean.append(pat.population)

        if l>ath:
            ath = l
            fn = '{}/ath_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,pat.population,int(l),args.seed,n)
            bb = pat.bounding_box
            pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
            log('ATH',n,k,l,ath,pat.population,m,r,nath)
            tree.show(idhidden=True)
            nath=0
            #init = pat.coords()
        else:
            #fn = '{}/lmax_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(args.results,pat.population,int(l),args.seed,n)
            #bb = pat.bounding_box
            #pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0],bb[1]), footer=None, comments=str(args), file_format='rle', save_comments=True)
            log('LMAX',n,k,l,ath,pat.population,m,r,nath)
        k=0
        lmax=l
        #bt+=args.btboost
        #bt+=np.log(lmax)
        #d = pat.digest()
        #if not any(h[2] == d for h in history):

#        for i in range(len(history)):
#            if history[i][0]==l:
#                history.pop(i)
#                break
        #history.append([l,pat.coords(),pat.digest()])
        #history.sort(reverse=True, key=lambda a: a[0]) # sort by lmax
#        vv=0
#        for v in tree.all_nodes():
#            if v.identifier==pat.digest():
#                vv+=1
#        if vv==0: # don't create dups
#            node = tree.create_node(tag=l, identifier=pat.digest(), parent=node, data=[pat.coords(),l])

#        d = pat.digest()
#        if d not in [v.identifier for v in tree.children(node.identifier)]: # no dups
#            node = tree.create_node(tag=l, identifier=d, parent=node, data=[pat.coords(),l])

#        d = pat.digest()
#        #if (l not in [v.tag for v in tree.children(node.identifier)]) and (d not in [v.identifier for v in tree.all_nodes()]): # no dups
#        if d not in [v.identifier for v in tree.all_nodes()]: # no dups
#            node = tree.create_node(tag=l, identifier=d, parent=node, data=[pat.coords(),l])

#        if d not in [v.identifier for v in tree.all_nodes()]: # no dups
#            node = tree.create_node(tag=l, identifier=d, parent=node, data=[pat.coords(),l])
#
#        if not any(h[1] == l for h in history):
#            history.append([pat.coords(),l])
#            history.sort(reverse=True, key=lambda a: a[1]) # sort by lmax
    else:
        pat[xy] = v0 # revert mutation xy
