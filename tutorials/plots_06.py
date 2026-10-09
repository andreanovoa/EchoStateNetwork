"""Architecture diagram of an `EchoStateNetwork`: a matplotlib port of our TikZ
schematic of the parallel network, with Paul Tol's vibrant colours, the same
reservoir glyph, and field strips whose cells are hatched by the reservoir that
reads (input) or predicts (output) them."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path

INP, OUT, RES, PAR, PAD = '#0077BB', '#009988', '#787878', '#EE7733', '#BBBBBB'
# independent patches: one colour per network (Tol muted)
NET = ['#CC6677', '#332288', '#DDAA33', '#117733', '#882255', '#44AA99', '#999933', '#AA4499']
HATCHES = ['////', '\\\\\\\\', '....', '||||', '----', 'oo']
CELL, WIDTH = 0.3, 0.48                       # one site of a strip, as in the TikZ schematic
X_IN, X_RES, X_OUT = 0.0, 3.0, 5.7

# reservoir glyph of the TikZ schematic, on the unit circle: 8 neurons, 10 edges, 3 self-loops
_NEURONS = 0.88 / 1.25 * np.array([(-0.24, 0.98), (0.66, 0.74), (-0.96, 0.34), (-0.1, 0.18),
                                   (1.02, -0.06), (-0.7, -0.6), (0.38, -0.38), (0, -1.02)])
_EDGES = [(1, 4), (2, 1), (3, 4), (4, 7), (5, 2), (7, 5), (6, 3), (4, 6), (8, 6), (7, 8)]
_LOOPS = [(4, 77.5), (7, -7.5), (3, 82.5)]
_SYMBOL = {'rho': r'$\rho$', 'sigma_in': r'$\sigma_\mathrm{in}$', 'tikh': 'tikh',
           'leak_rate': r'$\alpha$'}


def _y(row):
    return -row * CELL


def _arrow(ax, p0, p1, color, lw=1.8, head=11, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=head, color=color,
                                 lw=lw, shrinkA=0, shrinkB=0, **kw))


def _route(ax, points, color, lw=1.8, head=11, **kw):
    """Arrow along a polyline whose segments are horizontal or vertical."""
    path = Path(points, [Path.MOVETO] + [Path.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=head, color=color,
                                 lw=lw, joinstyle='miter', **kw))


def _brace(ax, x, y0, y1, color, width=0.12):
    """Curly brace from y0 to y1 at x, tip at x + width (width < 0: tip to the left)."""
    ym = 0.5 * (y0 + y1)
    path = Path([(x, y0), (x + width, y0), (x, ym), (x + width, ym),
                 (x, ym), (x + width, y1), (x, y1)], [Path.MOVETO] + [Path.CURVE4] * 6)
    ax.add_patch(PathPatch(path, fc='none', ec=color, lw=1.6))
    return x + width, ym


def _reservoir(ax, x, y, radius, color=RES):
    ax.add_patch(Circle((x, y), radius, fc='white', ec='k', lw=1.2, zorder=3))
    if radius < 0.3:                     # ponytail: too small for the glyph, plain circle
        return
    pts, dot = np.array([x, y]) + radius * _NEURONS, 0.07 * radius
    for a, b in _EDGES:
        p0, p1 = pts[a - 1], pts[b - 1]
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        _arrow(ax, p0 + 1.6 * dot * u, p1 - 1.6 * dot * u, color, lw=0.8, head=6 * radius, zorder=4)
    for i, o in _LOOPS:
        p, d = pts[i - 1], 0.46 * radius
        c1 = p + d * np.array([np.cos(np.radians(o)), np.sin(np.radians(o))])
        c2 = p + d * np.array([np.cos(np.radians(o - 75)), np.sin(np.radians(o - 75))])
        ax.add_patch(FancyArrowPatch(path=Path([p, c1, c2, p], [Path.MOVETO] + [Path.CURVE4] * 3),
                                     arrowstyle='-|>', mutation_scale=6 * radius, color=color,
                                     lw=0.8, zorder=4))
    for p in pts:
        ax.add_patch(Circle(p, dot, color='k', zorder=5))


def _cell(ax, x, row, color, hatches=(), dashed=False, text=None):
    y = _y(row)
    ax.add_patch(Rectangle((x - WIDTH / 2, y - CELL / 2), WIDTH, CELL, fc='white', ec=color,
                           lw=1.4, ls='--' if dashed else '-', zorder=1))
    for h in hatches:
        ax.add_patch(Rectangle((x - WIDTH / 2 + 0.02, y - CELL / 2 + 0.02), WIDTH - 0.04,
                               CELL - 0.04, fill=False, hatch=h, ec=color, lw=0, zorder=1))
    if text:
        ax.text(x - WIDTH / 2 - 0.06, y, text, ha='right', va='center', fontsize=7, color=color)


def _range_text(name, esn):
    lo_hi = getattr(esn, f'{name}_range')
    if name == 'tikh':
        return 'tikh in {' + ', '.join(f'{t:.0e}' for t in lo_hi) + '}'
    if name == 'sigma_in':
        return rf'$\sigma_\mathrm{{in}} \in 10^{{[{lo_hi[0]:g},\,{lo_hi[1]:g}]}}$'
    return f'{_SYMBOL[name]}' + rf' $\in [{lo_hi[0]:g},\,{lo_hi[1]:g}]$'


def describe(esn, strategy='RVC_Noise'):
    """Equations, matrix shapes and hyperparameter search of `esn`, one line each."""
    par, ind = esn.patch_size is not None, esn.patch_size is not None and not esn.shared
    n_param = esn.N_dim_in - len(esn.observed_idx)
    p, q = (r'^{(p)}_{' if par else r'_{'), (r'^{(p)}' if ind else '')   # patch: superscript (p)
    u_in = r'\tilde{\mathbf{u}}_n[\mathcal{I}^{(p)}]' if par else r'\tilde{\mathbf{u}}_n'
    if n_param and esn.param_in_reservoir:
        u_in += r';\, \tilde{\mathbf{p}}'
    a, s, rho = (r'\alpha^{(p)}', r'\sigma_\mathrm{in}^{(p)}', r'\rho^{(p)}') if ind else (r'\alpha', r'\sigma_\mathrm{in}', r'\rho')
    p_t = r';\, \tilde{\mathbf{p}}'
    read = {False: '', True: r';\, \tilde{\mathbf{u}}_n' + (p_t if n_param else ''), 'params': p_t}[esn.readout_input]
    out = r'\hat{\mathbf{u}}_{n+1}[\mathcal{P}^{(p)}]' if par else r'\hat{\mathbf{u}}_{n+1}'
    lines = [rf'$\mathbf{{r}}{p}n+1}} = (1-{a})\,\mathbf{{r}}{p}n}} + {a}\,\tanh({s}\,\mathbf{{W}}_\mathrm{{in}}{q}'
             rf'[{u_in};\, b_\mathrm{{in}}] + {rho}\,\mathbf{{W}}{q}\,\mathbf{{r}}{p}n}})$',
             rf'${out} = \mathbf{{W}}_\mathrm{{out}}^{{{q[1:] if q else ""}\top}}'
             rf'[\mathbf{{r}}{p}n+1}}{read};\, b_\mathrm{{out}}]$']
    times = f'  (x {esn.N_patches} networks)' if ind else ''
    lines.append(rf'$\mathbf{{W}}_\mathrm{{in}}$ {esn.N_units}$\times${esn._n_in + 1},  '
                 rf'$\mathbf{{W}}$ {esn.N_units}$\times${esn.N_units},  '
                 rf'$\mathbf{{W}}_\mathrm{{out}}$ {esn._n_readout}$\times${esn._n_out}{times};  '
                 rf'$\mathbf{{r}}$: {esn.N_r} rows')
    hp = list(esn.hyperparameters_to_optimize)
    fixed = ', '.join(f'{_SYMBOL[n]} = {getattr(esn, n):g}' for n in _SYMBOL if n not in hp)
    if hp:
        search = '; one search per patch' if ind else ''
        lines.append(f'Bayesian optimization ({strategy}{search}): '
                     + ', '.join(_range_text(n, esn) for n in hp))
    lines.append(f'fixed: {fixed}' if fixed else 'all hyperparameters optimized')
    return lines


def draw_esn(esn, ax=None, strategy='RVC_Noise', title=None):
    """Draw the architecture of `esn` (single reservoir or parallel layout) with
    its update equations, matrix shapes and hyperparameter search underneath."""
    esn._n_readout                # ValueError for readout_input='params' without input_parameters
    N, par = esn.N_dim, esn.patch_size is not None
    G, H, P = (esn.patch_size, esn.halo, esn.N_patches) if par else (N, 0, 1)
    ind = par and not esn.shared
    observed = set(np.asarray(esn.observed_idx).tolist())
    n_param = esn.N_dim_in - len(observed)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 0.32 * (N + 2 * H + n_param) + 3.2))

    # input strip: sites, periodic copies (dashed) or zero padding (grey), parameters on top
    windows = [p * G + np.arange(-H, G + H) for p in range(P)]
    reads = {}                                   # row -> hatches of the reservoirs that read it
    for p, rows in enumerate(windows):
        for r in rows:
            reads.setdefault(r, []).append(HATCHES[p % len(HATCHES)] if par else None)
    sites = esn.patch_sites if par else None
    for r in sorted(reads):
        hatches = [h for h in reads[r] if h]
        if 0 <= r < N:
            obs = r in observed
            _cell(ax, X_IN, r, INP if obs else PAD, hatches if obs else (), dashed=not obs)
        else:                                    # outside the domain
            k, c = (r - windows[0][0], 0) if r < 0 else (r - windows[-1][0], P - 1)
            site = sites[c, k]
            if site >= 0:
                _cell(ax, X_IN, r, INP, hatches, dashed=True, text=f'$x_{{{site}}}$')
            else:
                _cell(ax, X_IN, r, PAD, dashed=True, text='0')
    param_rows = [-1.5 - j for j in range(n_param)]
    for j, r in enumerate(param_rows):
        _cell(ax, X_IN, r, PAR, text=rf'$p_{{{n_param - 1 - j}}}$')
    top = min(min(reads), *param_rows) if param_rows else min(reads)
    ax.text(X_IN, _y(top) + 0.35, r'$\tilde{\mathbf{u}}_n$', ha='center', color=INP, fontsize=11)

    # output strip: every site, hatched by the reservoir that predicts it
    for r in range(N):
        _cell(ax, X_OUT, r, OUT, [HATCHES[(r // G) % len(HATCHES)]] if par else ())
    ax.text(X_OUT, _y(min(0, top)) + 0.35, r'$\hat{\mathbf{u}}_{n+1}$', ha='center', color=OUT,
            fontsize=11)

    # reservoirs, input windows and readouts
    radius = min(0.9, 0.42 * G * CELL)
    stagger = 3 if H else 1                      # ponytail: brace columns, enough while halo <= patch_size
    for p, rows in enumerate(windows):
        y_res = _y(p * G + (G - 1) / 2)
        x_tip, y_tip = _brace(ax, X_IN + WIDTH / 2 + 0.06 + 0.2 * (p % stagger),
                              _y(rows[-1]) - CELL / 2, _y(rows[0]) + CELL / 2, INP)
        color = NET[p % len(NET)] if ind else RES
        x_res = X_RES
        _arrow(ax, (x_tip, y_tip), (x_res - radius, y_res), INP)
        sup = f'^{{({p})}}' if ind else ''
        ax.text(0.5 * (x_tip + x_res - radius), y_res - 0.08, rf'$\mathbf{{W}}_\mathrm{{in}}{sup}$',
                ha='center', va='top', fontsize=8, color=INP, zorder=6,
                bbox=dict(fc='white', ec='none', pad=0.5))
        x_otip, y_otip = _brace(ax, X_OUT - WIDTH / 2 - 0.06, _y(p * G + G - 1) - CELL / 2,
                                _y(p * G) + CELL / 2, OUT, width=-0.12)
        _arrow(ax, (x_res + radius, y_res), (x_otip, y_otip), OUT)
        ax.text(0.5 * (x_res + radius + x_otip), y_res - 0.08, rf'$\mathbf{{W}}_\mathrm{{out}}{sup}$',
                ha='center', va='top', fontsize=8, color=OUT, zorder=6,
                bbox=dict(fc='white', ec='none', pad=0.5))
        _reservoir(ax, x_res, y_res, radius, color)
        ax.text(x_res + 0.7 * radius, y_res + 0.75 * radius, rf'$\mathbf{{r}}^{{({p})}}$' if par
                else r'$\mathbf{r}$', fontsize=9, color=color if ind else 'k', zorder=6)
        alpha = esn.patches[p].leak_rate if ind and esn.patches else esn.leak_rate
        if alpha < 1:                            # leaky integrator: memory of the previous state
            dx = 0.45 * radius
            yc, yt = y_res + np.sqrt(radius ** 2 - dx ** 2), y_res + 1.28 * radius
            _route(ax, [(x_res - dx, yc), (x_res - dx, yt), (x_res + dx, yt), (x_res + dx, yc)], 'k',
                   lw=1, head=7, zorder=6)
            ax.text(x_res + dx + 0.04, 0.5 * (yc + yt), r'$1-\alpha$', va='center', fontsize=7)

    # single reservoir only (the parallel layout rejects these): parameters enter from above,
    # right then down, at a junction of the input arrow and/or into the readout arrow; the
    # input skip leaves the input arrow down, right, up into the readout arrow
    bottom = _y(max(max(reads), N - 1)) - CELL / 2
    x_join = x_otip - 0.3
    if n_param:
        x_p, y_p = _brace(ax, X_IN + WIDTH / 2 + 0.06, _y(param_rows[0]) - CELL / 2,
                          _y(param_rows[-1]) + CELL / 2, PAR)
        if esn.param_in_reservoir:
            x_junc = x_tip + 0.3
            _route(ax, [(x_p, y_p), (x_junc, y_p), (x_junc, y_res + 0.07)], PAR)
            ax.plot(x_junc, y_res, 'D', ms=5, mfc='white', mec='k', mew=1.2, zorder=7)
        if esn.readout_input == 'params' or (esn.readout_input is True and not esn.param_in_reservoir):
            _route(ax, [(x_p, y_p), (x_join, y_p), (x_join, y_res)], PAR)
            ax.text(X_RES, y_p + 0.05, r'$\mathbf{W}_\mathrm{out}$ reads $\tilde{\mathbf{p}}$',
                    ha='center', va='bottom', fontsize=8, color=PAR)
    if esn.readout_input is True:
        x_b, y_low = x_tip + 0.5, y_res - radius - 0.3
        _route(ax, [(x_b, y_res), (x_b, y_low), (x_join, y_low), (x_join, y_res)], INP)
        carried = r';\,\tilde{\mathbf{p}}' if n_param and esn.param_in_reservoir else ''
        ax.text(0.5 * (x_b + x_join), y_low - 0.06, r'input skip: $\mathbf{W}_\mathrm{out}$ reads '
                rf'$[\mathbf{{r}};\,\tilde{{\mathbf{{u}}}}_n{carried}]$', ha='center', va='top',
                fontsize=8, color=INP)
        bottom = min(bottom, y_low - 0.3)

    # closed loop: the prediction (its observed components) is the next input
    y_loop = bottom - 0.6
    _route(ax, [(X_OUT, _y(N - 1) - CELL / 2), (X_OUT, y_loop), (X_IN, y_loop),
                (X_IN, _y(max(reads)) - CELL / 2)], INP)
    loop = 'closed loop' + (': observed components' if len(observed) < N else '')
    ax.text(X_RES, y_loop - 0.08, loop, ha='center', va='top', fontsize=8, color=INP)

    lines = describe(esn, strategy)
    for k, line in enumerate(lines):
        ax.text(X_IN - 0.9, y_loop - 0.6 - 0.36 * k, line, fontsize=8, va='top')
    if title:
        ax.set_title(title, loc='left', fontsize=10)
    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.set_ylim(y_loop - 0.7 - 0.36 * len(lines), ax.get_ylim()[1] + 0.7)   # room for the strip labels
    ax.axis('off')
    return ax
