import argparse
import random
import datetime
import time
import numpy as np ; print('numpy ' + np.__version__)
import lifelib ; print('lifelib',lifelib.__version__)
import os
import glob
import re
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

np.set_printoptions(linewidth=250)

# ---------------------------------------------------------------------------
# Module-level worker state — populated by _worker_init in each subprocess
# ---------------------------------------------------------------------------
_w_lt      = None
_w_gliders = None
_w_spacex  = None
_w_spacey  = None

def _worker_init(memory_mb, gliders_list, spacex, spacey):
    global _w_lt, _w_gliders, _w_spacex, _w_spacey
    import lifelib as _ll, numpy as _np
    _sess   = _ll.load_rules("b3s23")
    _w_lt   = _sess.lifetree(memory=memory_mb)
    _w_gliders = gliders_list
    _w_spacex  = spacex
    _w_spacey  = spacey

def _worker_eval(task):
    """Evaluate one candidate pattern. Returns (l, pop, rle_str, bb) where
    l>=0 is stable gen, l==-1 exception, l==-2 linear growth."""
    vec, parent_lmax, step, period = task
    import numpy as _np

    pat = _w_lt.pattern()

    # --- render vec onto pat ---
    if isinstance(vec, tuple):          # (rle_str, dict) from --rleinit
        rle_str, vec_dict = vec
        base = _w_lt.pattern(rle_str)
        bb = base.bounding_box
        if bb is not None:
            rx, ry, rw, rh = bb
            pat[rx:rx+rw, ry:ry+rh] |= base[rx:rx+rw, ry:ry+rh]
    else:
        vec_dict = vec

    for key, s in vec_dict.items():
        x0 = int(key[0]) * _w_spacex
        y0 = int(key[1]) * _w_spacey
        xy = [[x0+x, y0+y] for (x, y) in _w_gliders[s] if x is not None]
        if xy:
            pat[_np.array(xy)] |= 1

    # --- save initial bbox for linear-growth detection ---
    bb0 = pat.bounding_box
    if bb0 is not None:
        cx0   = bb0[0] + bb0[2] // 2
        cy0   = bb0[1] + bb0[3] // 2
        half0 = max(bb0[2], bb0[3]) * 3 + 200
        bbox_thresh = max(bb0[2], bb0[3]) * 10 + 500
    else:
        cx0 = cy0 = 0; half0 = 1000; bbox_thresh = 5000

    # --- HashLife jump to near parent_lmax ---
    o = int(max(0, parent_lmax - step * period))
    pat = pat.advance(o)

    max_windows = 150000 // (step * period) + 2
    gen, final_pop, final_pat = pat.advance_until_stable(
        step=step, period=period, max_windows=max_windows)

    if gen >= 0:
        return max(0, o + gen), final_pop, None, None

    # --- never stabilised: check for linear growth ---
    bb = final_pat.bounding_box
    if bb is not None and max(bb[2], bb[3]) > bbox_thresh:
        pop_total = final_pat.population
        if pop_total > 0:
            pop_inside = final_pat[cx0-half0:cx0+half0, cy0-half0:cy0+half0].population
            if (pop_total - pop_inside) / pop_total > 0.15:
                try:
                    rle_str = final_pat.rle_string()
                    rle_bb  = list(bb)
                except Exception:
                    rle_str = None; rle_bb = None
                return -2, pop_total, rle_str, rle_bb

    return -1, final_pop, None, None


# ---------------------------------------------------------------------------
# Pool node — forms an explicit parent→child tree
# ---------------------------------------------------------------------------
class PoolNode:
    __slots__ = ('nid', 'lifespan', 'vec', 'failures', 'parent', 'children', 'depth')

    def __init__(self, nid, lifespan, vec, failures=0, parent=None, depth=0):
        self.nid      = nid
        self.lifespan = lifespan
        self.vec      = vec
        self.failures = failures
        self.parent   = parent
        self.children = []
        self.depth    = depth


pool     = {}   # nid -> PoolNode
_next_nid = 0


def _pool_add(lifespan, vec, parent_nid=None):
    global _next_nid
    depth = 0
    if parent_nid is not None and parent_nid in pool:
        depth = pool[parent_nid].depth + 1
        pool[parent_nid].children.append(_next_nid)
    node = PoolNode(_next_nid, lifespan, vec, parent=parent_nid, depth=depth)
    pool[_next_nid] = node
    _next_nid += 1
    return node.nid


def _pool_remove(nid):
    if nid not in pool:
        return
    node = pool.pop(nid)
    for cid in node.children:
        if cid in pool:
            pool[cid].parent = None          # orphan children
    if node.parent is not None and node.parent in pool:
        p = pool[node.parent]
        p.children = [c for c in p.children if c != nid]


def _pool_prune(prune_limit):
    if not args.prune_lmin and not args.prune_cmax:
        return
    nodes = list(pool.values())
    while len(pool) > prune_limit:
        if args.prune_cmax:
            worst = max(nodes, key=lambda nd: nd.failures)
            _pool_remove(worst.nid)
            nodes = list(pool.values())
        if args.prune_lmin:
            worst = min(nodes, key=lambda nd: nd.lifespan)
            _pool_remove(worst.nid)
            nodes = list(pool.values())


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _vec_to_serialisable(vec):
    """Convert a pool vec for pickling (replace lifelib Pattern with rle_str)."""
    if isinstance(vec, tuple):
        rle_pat, d = vec
        # rle_pat might already be a string if loaded from checkpoint
        if isinstance(rle_pat, str):
            return ('rle', rle_pat, d)
        return ('rle', rle_pat.rle_string(), d)
    return vec


def _vec_from_serialisable(sv):
    if isinstance(sv, tuple) and sv[0] == 'rle':
        return (sv[1], sv[2])   # (rle_str, dict)
    return sv


def save_checkpoint(path, state):
    data = {
        'pool': [{'nid': nd.nid, 'lifespan': nd.lifespan,
                  'vec': _vec_to_serialisable(nd.vec), 'failures': nd.failures,
                  'parent': nd.parent, 'children': list(nd.children),
                  'depth': nd.depth}
                 for nd in pool.values()],
        'next_nid': _next_nid,
        **state,
    }
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    os.replace(tmp, path)
    print('CHKPT  saved {} nodes n={}'.format(len(pool), state.get('n', 0)), flush=True)


def load_checkpoint(path):
    global pool, _next_nid
    with open(path, 'rb') as f:
        data = pickle.load(f)
    pool.clear()
    for nd in data['pool']:
        node = PoolNode(nd['nid'], nd['lifespan'], _vec_from_serialisable(nd['vec']),
                        nd['failures'], nd['parent'], nd.get('depth', 0))
        node.children = list(nd['children'])
        pool[nd['nid']] = node
    _next_nid = data['next_nid']
    print('CHKPT  loaded {} nodes next_nid={}'.format(len(pool), _next_nid), flush=True)
    return data


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--p16',        default=False, action='store_true')
parser.add_argument('--figeight',   default=False, action='store_true')
parser.add_argument('--blank',      default=False, action='store_true')
parser.add_argument('--traffic',    default=False, action='store_true')
parser.add_argument('--weighted',   default=False, action='store_true')
parser.add_argument('--delete',     default=False, action='store_true')
parser.add_argument('--ttet',       default=False, action='store_true')
parser.add_argument('--tub',        default=False, action='store_true')
parser.add_argument('--pihep',      default=False, action='store_true')
parser.add_argument('--squeeze',    help='cmean coefficient', default=1, type=float)
parser.add_argument('--bipolar',    default=16, type=int)
parser.add_argument('--xrad',       help='gaussian radius', default=1, type=float)
parser.add_argument('--yrad',       help='gaussian radius', default=1, type=float)
parser.add_argument('--spark',      default=False, action='store_true')
parser.add_argument('--nspark',     help='number of initial sparks in tree', default=1, type=int)
parser.add_argument('--ispark',     help='number of sparks in each nspark', default=1, type=int)
parser.add_argument('--mutalgo',    help='ternary, remove', default='ternary')
parser.add_argument('--checkpoint', help='log interval (seconds)', default=10000000000, type=int)
parser.add_argument('--space',      help='grid spacing', default=3, type=int)
parser.add_argument('--meth',       default=False, action='store_true')
parser.add_argument('--rpent',      default=False, action='store_true')
parser.add_argument('--glider',     default=False, action='store_true')
parser.add_argument('--bigblock',   default=False, action='store_true')
parser.add_argument('--blinker',    default=False, action='store_true')
parser.add_argument('--block',      default=False, action='store_true')
parser.add_argument('--radalgo',    help='radius adjustment', default='const')
parser.add_argument('--hysteresis', help='prune fraction', default=0.5, type=float)
parser.add_argument('--best',       default=False, action='store_true')
parser.add_argument('--irad',       help='initial radius', default=10, type=int)
parser.add_argument('--fade',       help='mutate delete glider rate', default=0.5, type=float)
parser.add_argument('--nullstate',  default=False, action='store_true')
parser.add_argument('--fast',       default=False, action='store_true')
parser.add_argument('--init',       help='initial lattice', default='grid')
parser.add_argument('--mutfreq',    help='mutation frequency {single,expo}', default='expo')
parser.add_argument('--linit',      help='minimum lifespan for initial seeds', default=0, type=int)
parser.add_argument('--mutset',     help='mutation disjoint sets {single,dual}', default='single')
parser.add_argument('--dup',        default=False, action='store_true')
parser.add_argument('--reseed',     default=False, action='store_true')
parser.add_argument('--ipool',      help='minimum samples before pruning', default=100, type=int)
parser.add_argument('--prune',      help='prune threshold', default=20, type=float)
parser.add_argument('--prune_lmin', help='prune lowest-lifespan node when pool exceeds --prune',
                    default=False, action='store_true')
parser.add_argument('--prune_cmax', help='prune highest-failure-count node when pool exceeds --prune',
                    default=False, action='store_true')
parser.add_argument('--rad',        help='gaussian radius', default=0, type=float)
parser.add_argument('--n',          help='gaussian gliders', default=10, type=int)
parser.add_argument('--spacex',     help='grid spacing', default=10, type=int)
parser.add_argument('--spacey',     help='grid spacing', default=10, type=int)
parser.add_argument('--sidex',      help='grid size = side*2+1', default=50, type=int)
parser.add_argument('--sidey',      help='grid size = side*2+1', default=50, type=int)
parser.add_argument('--seed',       help='random seed', default=None, type=int)
parser.add_argument('--memory',     help='GC limit per lifetree in MB (total split across workers)', default=50000, type=int)
parser.add_argument('--results',    help='results directory', default='./results')
parser.add_argument('--rleinit',    help='directory of .rle files to seed initial pool', default=None)
parser.add_argument('--debug',      default=False, action='store_true')
# --- new performance / parallelism args ---
parser.add_argument('--nworkers',   help='parallel worker processes (1=serial)', default=1, type=int)
parser.add_argument('--step',       help='advance step size for advance_until_stable', default=1, type=int)
# --- checkpoint args ---
parser.add_argument('--chkpt_save',     help='checkpoint file path', default=None)
parser.add_argument('--chkpt_interval', help='checkpoint interval (seconds)', default=3600, type=int)
parser.add_argument('--load',           help='resume from checkpoint file', default=None)
args = parser.parse_args()

if args.seed is None:
    args.seed = random.randint(1, 1000000)
random.seed(args.seed)
np.random.seed(args.seed)
print(args)

sess = lifelib.load_rules("b3s23")
lt   = sess.lifetree(memory=args.memory)

if not os.path.exists(args.results):
    os.makedirs(args.results)
logf = open('results/log.{}'.format(datetime.datetime.now().isoformat()), 'w')
print('ARGS', args, file=logf)


def log(hdr):
    nodes = list(pool.values())
    larr  = np.array([nd.lifespan  for nd in nodes]) if nodes else np.array([0])
    carr  = np.array([nd.failures  for nd in nodes]) if nodes else np.array([0])
    msg = ('{:10} wall {} n {:6d} k {:6d} LIFE {:9.0f} ath {:9.0f} pop {:4d} m {:4d}'
           ' prune {:8.0f} node {:6d} final {:6d} lprev {:6.0f} cmax {:6.0f}'
           ' lmean {:9.0f} uniq {:9.0f} lmax {:9.0f} cmean {:6.2f} lmin {:9.3f}'
           ' rad {:6.2f} depth {:4d}').format(
        hdr, datetime.datetime.now(), n, n1, l, ath, _log_pop, m, prune,
        len(pool), pop, lprev, np.max(carr), np.mean(larr), len(uniq),
        np.max(larr), np.mean(carr), np.min(larr), rad, _log_depth)
    print(msg)
    print(msg, file=logf)
    logf.flush()


# ---------------------------------------------------------------------------
# Glider / primitive definitions
# ---------------------------------------------------------------------------
def transform_coords(coords, N=9):
    transform_map = {
        "identity":     lambda x, y: (x, y),
        "flip":         lambda x, y: (N - x, y),
        "flip_x":       lambda x, y: (N - x, y),
        "flip_y":       lambda x, y: (x, N - y),
        "transpose":    lambda x, y: (y, x),
        "swap_xy":      lambda x, y: (y, x),
        "rot90":        lambda x, y: (y, N - x),
        "rot180":       lambda x, y: (N - x, N - y),
        "rot270":       lambda x, y: (N - y, x),
        "rcw":          lambda x, y: (y, N - x),
        "rccw":         lambda x, y: (N - y, x),
        "swap_xy_flip": lambda x, y: (N - y, N - x),
    }
    results = {}
    for name, func in transform_map.items():
        results[name] = [func(x, y) for x, y in coords]
    return results


gliders = []
if args.p16:
    on_cells = [(0,5),(0,6),(1,2),(1,5),(1,6),(2,1),(2,6),(3,1),(3,6),(4,6),
                (5,0),(5,1),(5,3),(5,5),(6,0),(6,1),(6,3),(6,4),(8,5),(8,6),(8,7)]
    coords = transform_coords(on_cells)
    for t in coords:
        gliders.append(coords[t])

if args.blank:
    for _ in range(1):
        gliders.append(np.array([[None, None]]))

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
        gliders.append(np.array([[-3,-1],[-3,0],[-3,1],[-1,3],[0,3],[1,3],
                                  [3,-1],[3,0],[3,1],[-1,-3],[0,-3],[1,-3]]))

if args.figeight:
    three = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]
    gliders.append(np.concatenate([three, [(3+x,3+y) for (x,y) in three]], axis=0))
    gliders.append(np.concatenate([[(3+x,0+y) for (x,y) in three],
                                    [(0+x,3+y) for (x,y) in three]], axis=0))
    gliders.append(np.array([[None, None]]))

if args.bigblock:
    for _ in range(8):
        gliders.append(np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]))

if args.blinker:
    for _ in range(4):
        gliders.append(np.array([[0,-1],[0,0],[0,1]]))
        gliders.append(np.array([[-1,0],[0,0],[1,0]]))

if args.block:
    for _ in range(2):
        gliders.append(np.array([[0,0],[0,1],[1,0],[1,1]]))
        gliders.append(np.array([[-1,0],[-1,1],[0,0],[0,1]]))
        gliders.append(np.array([[0,-1],[0,0],[1,-1],[1,0]]))
        gliders.append(np.array([[-1,-1],[-1,0],[0,-1],[0,0]]))

if args.rpent:
    gliders.append(np.array([[0,-1],[0,0],[0,1],[-1,0],[1,1]]))
    gliders.append(np.array([[-1,0],[0,0],[1,0],[0,1],[1,-1]]))
    gliders.append(np.array([[0,1],[0,0],[0,-1],[1,0],[-1,-1]]))
    gliders.append(np.array([[1,0],[0,0],[-1,0],[0,-1],[-1,1]]))
    gliders.append(np.array([[0,1],[0,0],[0,-1],[-1,0],[1,-1]]))
    gliders.append(np.array([[0,-1],[0,0],[0,1],[1,0],[-1,1]]))
    gliders.append(np.array([[-1,0],[0,0],[1,0],[0,-1],[1,1]]))
    gliders.append(np.array([[1,0],[0,0],[-1,0],[0,1],[-1,-1]]))

if args.meth:
    coords = np.array([[-1,0],[1,0],[0,1],[-1,1],[-2,-1],[1,-1],[2,-1]])
    gliders.append(coords)
    gliders.append(np.array([[y,-x] for x,y in coords]))
    gliders.append(np.array([[-x,-y] for x,y in coords]))
    gliders.append(np.array([[-y,x] for x,y in coords]))
    gliders.append(np.array([[x,-y] for x,y in coords]))
    gliders.append(np.array([[-x,y] for x,y in coords]))
    gliders.append(np.array([[y,x] for x,y in coords]))
    gliders.append(np.array([[-y,-x] for x,y in coords]))

if args.pihep:
    coords = np.array([[-1,-1],[-1,0],[-1,1],[0,1],[1,-1],[1,0],[1,1]])
    gliders.append(coords)
    gliders.append(np.array([[y,-x] for x,y in coords]))
    gliders.append(np.array([[-x,-y] for x,y in coords]))
    gliders.append(np.array([[-y,x] for x,y in coords]))
    gliders.append(np.array([[x,-y] for x,y in coords]))
    gliders.append(np.array([[-x,y] for x,y in coords]))
    gliders.append(np.array([[y,x] for x,y in coords]))
    gliders.append(np.array([[-y,-x] for x,y in coords]))

if args.tub:
    coords = np.array([[0,1],[0,-1],[-1,0],[1,0]])
    gliders.append(coords)
    gliders.append(np.array([[y,-x] for x,y in coords]))
    gliders.append(np.array([[-x,-y] for x,y in coords]))
    gliders.append(np.array([[-y,x] for x,y in coords]))
    gliders.append(np.array([[x,-y] for x,y in coords]))
    gliders.append(np.array([[-x,y] for x,y in coords]))
    gliders.append(np.array([[y,x] for x,y in coords]))
    gliders.append(np.array([[-y,-x] for x,y in coords]))

if args.ttet:
    coords = np.array([[-1,0],[0,0],[1,0],[0,-1]])
    gliders.append(coords)
    gliders.append(np.array([[y,-x] for x,y in coords]))
    gliders.append(np.array([[-x,-y] for x,y in coords]))
    gliders.append(np.array([[-y,x] for x,y in coords]))
    gliders.append(np.array([[x,-y] for x,y in coords]))
    gliders.append(np.array([[-x,y] for x,y in coords]))
    gliders.append(np.array([[y,x] for x,y in coords]))
    gliders.append(np.array([[-y,-x] for x,y in coords]))

states = range(len(gliders))

# Serialisable copy of gliders for passing to worker subprocesses
_gliders_serial = [g.tolist() if hasattr(g, 'tolist') else g for g in gliders]


def render(pat, v):
    bb = pat.bounding_box
    if bb is not None:
        pat[bb[0]:bb[0]+bb[2], bb[1]:bb[1]+bb[3]] = 0
    if isinstance(v, tuple):
        base_rle, vec = v
        if isinstance(base_rle, str):
            base_pat = lt.pattern(base_rle)
        else:
            base_pat = base_rle
        rle_bb = base_pat.bounding_box
        if rle_bb is not None:
            rx, ry, rw, rh = rle_bb
            pat[rx:rx+rw, ry:ry+rh] |= base_pat[rx:rx+rw, ry:ry+rh]
    else:
        vec = v
    for key in vec.keys():
        s  = vec[key]
        x0 = key[0] * args.spacex
        y0 = key[1] * args.spacey
        xy = [[x0+x, y0+y] for (x, y) in gliders[s] if x is not None]
        xy = np.array(xy)
        pat[xy] |= 1


# ---------------------------------------------------------------------------
# In-process evaluation (serial mode, nworkers==1)
# ---------------------------------------------------------------------------
def _eval_inline(vec, parent_lmax):
    render(pat, vec)

    bb0 = pat.bounding_box
    if bb0 is not None:
        cx0   = bb0[0] + bb0[2] // 2
        cy0   = bb0[1] + bb0[3] // 2
        half0 = max(bb0[2], bb0[3]) * 3 + 200
        bbox_thresh = max(bb0[2], bb0[3]) * 10 + 500
    else:
        cx0 = cy0 = 0; half0 = 1000; bbox_thresh = 5000

    period = 100
    o = int(max(0, parent_lmax - args.step * period))
    p = pat.advance(o)

    max_windows = 150000 // (args.step * period) + 2
    gen, final_pop, final_pat = p.advance_until_stable(
        step=args.step, period=period, max_windows=max_windows)

    if gen >= 0:
        return max(0, o + gen), final_pop, None, None

    bb = final_pat.bounding_box
    if bb is not None and max(bb[2], bb[3]) > bbox_thresh:
        pop_total = final_pat.population
        if pop_total > 0:
            pop_inside = final_pat[cx0-half0:cx0+half0, cy0-half0:cy0+half0].population
            if (pop_total - pop_inside) / pop_total > 0.15:
                try:
                    rle_str = final_pat.rle_string()
                    rle_bb  = list(bb)
                except Exception:
                    rle_str = None; rle_bb = None
                return -2, pop_total, rle_str, rle_bb

    return -1, final_pop, None, None


# ---------------------------------------------------------------------------
# Initial pool construction
# ---------------------------------------------------------------------------
uniq  = []
pat   = lt.pattern()

# Logging globals (updated by the result-processing code)
n = n0 = n1 = 0
ath    = 0
lmax0  = None
m      = 0
prune  = args.prune
nbest  = 0
ntime  = time.time()
lprev  = 0
pop    = 0
l      = 0
rad    = args.rad
_log_pop   = 0
_log_depth = 0
last_chkpt = time.time()

if args.load:
    # ---- resume from checkpoint ----
    ck = load_checkpoint(args.load)
    n      = ck.get('n', 0)
    ath    = ck.get('ath', 0)
    lmax0  = ck.get('lmax0', None)
    uniq   = [True] * ck.get('uniq_count', 0)
    n0 = n1 = 0
    print('Resumed: n={} ath={} pool_size={}'.format(n, ath, len(pool)))
else:
    # ---- build initial pool from scratch ----
    if args.rleinit:
        rle_files = sorted(f for f in glob.glob(os.path.join(args.rleinit, '*.rle'))
                           if not any(t in os.path.basename(f)
                                      for t in ('runaway','linear','exception')))
        print('RLEINIT: loading {} files from {}'.format(len(rle_files), args.rleinit))
        for k, fn in enumerate(rle_files):
            with open(fn, 'r') as f:
                rle_text = f.read()
            rle_pat = lt.pattern(rle_text)
            import re as _re
            mm = _re.search(r'_L(\d+)_', os.path.basename(fn))
            lhint = int(mm.group(1)) if mm else 0
            vec   = (rle_text, {})
            lv, pv, _, _ = _eval_inline(vec, lhint)
            print('RLEINIT k {:4d} fn {} lhint {} l {} pop {}'.format(
                k, os.path.basename(fn), lhint, lv, rle_pat.population))
            if lv > 0:
                _pool_add(lv, vec)
        print('RLEINIT: {} patterns added'.format(len(pool)))
    else:
        for k in range(args.nspark):
            while True:
                vec = {}
                for _ in range(1 + np.random.randint(args.ispark)):
                    vec[(np.floor(np.random.uniform(-args.irad, args.irad)),
                         np.floor(np.random.uniform(-args.irad, args.irad)))] = random.choice(states)
                lv, pv, _, _ = _eval_inline(vec, 0)
                if lv <= 0:
                    continue
                bb = pat.bounding_box
                if k < 10 and bb is not None:
                    fn = '{}/init_K{:06d}_seed{:09d}.rle'.format(args.results, k, args.seed)
                    pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0], bb[1]),
                                  footer=None, comments=str(args), file_format='rle',
                                  save_comments=True)
                print('INIT k', k, 'l', lv, 'pop', pat.population)
                _pool_add(lv, vec)
                break

# Initialise lmax0 to the pool's current best so the main loop only fires
# LMAX when the pool max actually increases beyond what seeding already found.
if pool:
    lmax0 = max(nd.lifespan for nd in pool.values())


# ---------------------------------------------------------------------------
# Main evolutionary loop
# ---------------------------------------------------------------------------
def _pick_parent():
    nodes = list(pool.values())
    larr  = np.array([nd.lifespan for nd in nodes])
    if args.weighted:
        s = np.sum(larr)
        prob = larr / s if s > 0 else None
        idx  = np.random.choice(len(nodes), p=prob)
    else:
        idx = np.random.choice(len(nodes))
    return nodes[idx]


def _make_mutation(node):
    global m
    ivec = node.vec
    if isinstance(ivec, tuple):
        imut_base, imut_vec = ivec[0], ivec[1].copy()
    else:
        imut_base = None
        imut_vec  = ivec.copy()

    m = int(np.ceil(random.expovariate(1)))
    action = 'add'
    for _ in range(m):
        if args.delete:
            action = random.choice(['add', 'del'])
        else:
            action = 'add'
        min_vec = 0 if imut_base is not None else 1
        if action == 'del' and len(imut_vec) > min_vec:
            key = random.choice(list(imut_vec.keys()))
            del imut_vec[key]
        else:
            dx = np.random.normal(0, rad)
            dy = np.random.normal(0, rad)
            imut_vec[(int(dx), int(dy))] = random.choice(states)

    return (imut_base, imut_vec) if imut_base is not None else imut_vec, action


def _process_result(l_res, pop_res, rle_str, rle_bb, parent_nid, imut, parent_lmax, action):
    global n, n0, n1, ath, lmax0, nbest, ntime, pat, prune, lprev, l, pop, _log_pop, _log_depth, last_chkpt

    n    += 1
    l     = l_res
    pop   = pop_res

    # render on main process for logging / RLE saving
    render(pat, imut)
    _log_pop   = pat.population
    _log_depth = pool[parent_nid].depth + 1 if parent_nid in pool else 0

    if l == -2:
        if parent_nid in pool:
            pool[parent_nid].failures += 1
        if rle_str is not None and rle_bb is not None:
            gen_approx = int(max(0, parent_lmax) + 150000)
            fn = '{}/linear_L{:09d}_seed{:09d}_n{:09d}.rle'.format(
                args.results, gen_approx, args.seed, n)
            with open(fn, 'w') as f:
                f.write('#CXRLE Pos={},{}\n'.format(rle_bb[0], rle_bb[1]))
                f.write(rle_str)
        log('LINEAR')

    elif l == -1:
        if parent_nid in pool:
            pool[parent_nid].failures += 1
        log('EXCEPTION')

    elif l > parent_lmax:
        lprev = parent_lmax
        n1 = n - n0; n0 = n
        uniq.append(True)
        new_nid = _pool_add(l, imut, parent_nid)
        _log_depth = pool[new_nid].depth
        _pool_prune(int(prune))

        # lmax pool update: fire only when pool max strictly increases.
        # Use a separate pattern object so pat (current mutation) is not clobbered.
        nodes = list(pool.values())
        cur_pool_max = max(nd.lifespan for nd in nodes)
        if lmax0 is not None and cur_pool_max > lmax0:
            lmax0 = cur_pool_max
            best_node = max(nodes, key=lambda nd: nd.lifespan)
            lmax_pat = lt.pattern()
            render(lmax_pat, best_node.vec)
            bb_lmax = lmax_pat.bounding_box
            if bb_lmax is not None:
                fn = '{}/lmax_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(
                    args.results, lmax_pat.population, int(lmax0), args.seed, n)
                lmax_pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb_lmax[0], bb_lmax[1]),
                                    footer=None, comments=str(args), file_format='rle',
                                    save_comments=True)
            log('LMAX')

        # pat still holds the current mutation (rendered at top of _process_result)
        bb = pat.bounding_box
        if l > ath:
            ath = l
            if bb is not None:
                fn = '{}/ATH_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(
                    args.results, pat.population, int(l), args.seed, n)
                pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0], bb[1]),
                               footer=None, comments=str(args), file_format='rle',
                               save_comments=True)
            log('ATH')
        else:
            nbest += 1
            nt0 = time.time()
            if args.best or (nt0 - ntime) > args.checkpoint:
                ntime = nt0
                if bb is not None:
                    fn = '{}/BEST_P{:06d}_L{:06d}_seed{:09d}_n{:09d}.rle'.format(
                        args.results, pat.population, int(l), args.seed, n)
                    pat.write_rle(fn, header='#CXRLE Pos={},{}\n'.format(bb[0], bb[1]),
                                   footer=None, comments=str(args), file_format='rle',
                                   save_comments=True)
            log('BEST')

    else:
        if action != 'del' and parent_nid in pool:
            pool[parent_nid].failures += 1

    # periodic checkpoint
    if args.chkpt_save:
        now = time.time()
        if now - last_chkpt >= args.chkpt_interval:
            last_chkpt = now
            save_checkpoint(args.chkpt_save, {
                'n': n, 'ath': ath, 'lmax0': lmax0, 'uniq_count': len(uniq)})


try:
    if args.nworkers == 1:
        # ----- serial mode -----
        while True:
            if not pool:
                print('EMPTY POOL'); break

            nodes = list(pool.values())
            larr  = np.array([nd.lifespan for nd in nodes])
            carr  = np.array([nd.failures for nd in nodes])
            if args.radalgo == 'pi':
                rad = np.sqrt(np.mean(larr)) / np.pi / args.spacex
            else:
                rad = args.rad

            node = _pick_parent()
            imut, action = _make_mutation(node)
            l_res, pop_res, rle_str, rle_bb = _eval_inline(imut, node.lifespan)
            _process_result(l_res, pop_res, rle_str, rle_bb,
                            node.nid, imut, node.lifespan, action)

    else:
        # ----- parallel mode -----
        mem_per_worker = max(100, args.memory // args.nworkers)
        QUEUE_DEPTH    = args.nworkers * 2

        with ProcessPoolExecutor(
                max_workers=args.nworkers,
                initializer=_worker_init,
                initargs=(mem_per_worker, _gliders_serial, args.spacex, args.spacey)
        ) as executor:

            in_flight = {}   # future -> (parent_nid, imut, parent_lmax, action)

            def _submit_one():
                if not pool:
                    return
                nodes  = list(pool.values())
                larr_l = np.array([nd.lifespan for nd in nodes])
                if args.radalgo == 'pi':
                    global rad
                    rad = np.sqrt(np.mean(larr_l)) / np.pi / args.spacex
                node = _pick_parent()
                imut, action = _make_mutation(node)
                task = (imut, node.lifespan, args.step, 100)
                fut  = executor.submit(_worker_eval, task)
                in_flight[fut] = (node.nid, imut, node.lifespan, action)

            # pre-fill pipeline
            for _ in range(QUEUE_DEPTH):
                _submit_one()

            while True:
                if not pool:
                    print('EMPTY POOL'); break
                if not in_flight:
                    _submit_one(); continue

                # Wait for at least one completion, then drain all ready futures
                next(as_completed(in_flight))   # blocks until one is done
                done_futs = [f for f in in_flight if f.done()]
                for fut in done_futs:
                    parent_nid, imut, parent_lmax, action = in_flight.pop(fut)
                    try:
                        l_res, pop_res, rle_str, rle_bb = fut.result()
                    except Exception as e:
                        print('WORKER ERROR', e)
                        l_res, pop_res, rle_str, rle_bb = -1, 0, None, None
                    _process_result(l_res, pop_res, rle_str, rle_bb,
                                    parent_nid, imut, parent_lmax, action)
                    _submit_one()

except KeyboardInterrupt:
    print('\nSTOPPING')
    if args.chkpt_save:
        save_checkpoint(args.chkpt_save,
                        {'n': n, 'ath': ath, 'lmax0': lmax0, 'uniq_count': len(uniq)})
    exit()
