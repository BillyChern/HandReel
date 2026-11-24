"""Regenerate five tables of the paper source's generator from the records in `reproduce/records/` (CPU, seconds).

    python reproduce/make_released_tables.py --out <dir>

This file is derived from the paper source's table generator `tools/make_tables.py` (sha256
887a266cd73512cbf4e507c8b5d3d6fc69fd610d907a979237265ef1d1105d24, paper source revision 2b6e3f9348c6ae01d5516b761c8c2b344a40ff08).
It holds that generator's functions for the five tables whose records are all in `reproduce/records/` (`solve.tex`,
`optim.tex`, `stride.tex`, `hyper.tex` and `weak_rows.tex`) and the two caption helpers they use, copied with
`ast.unparse`. Their comments, docstrings and LaTeX comment lines were removed, and each call to `write` no longer
passes the provenance string, which names files that are not released. The records are read under their release
names (`release_path` in `reproduce/manifest.json`; `docs/record_names.md` maps them to the paper source's names),
the weak-rows table is written as `weak_rows.tex` (the paper source's file has another name; its LaTeX label is
unchanged), and local names and assertion messages that were internal study numbers are descriptive. `rec`, `write` and `main` at the end of this
file are new. Each table still asserts the values it prints, so a changed record stops the run. Each table it writes
starts with a two-line `%` header naming this file; `reproduce/regenerate_tables.py` compares the rest of the file,
without its `%` comment lines, with the paper generator's output at the pinned revision (none of the five is a
table of the paper's main text, whose tables the paper source writes by hand). The copied code keeps the generator's style, so ruff
does not check this file.
"""
# ruff: noqa
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REC = Path(__file__).resolve().parent / "records"
OUT = Path("tables")


def _brace_end(s, i):
    d = 0
    for j in range(i, len(s)):
        if s[j] in '{}' and s[j - 1] == '\\':
            continue
        if s[j] == '{':
            d += 1
        elif s[j] == '}':
            d -= 1
            if d == 0:
                return j + 1
    raise ValueError('unbalanced caption')


def caption_above(s):
    out, i = ([], 0)
    while True:
        m = re.search('\\\\begin\\{(table\\*?)\\}(\\[[^\\]]*\\])?[^\\n]*\\n', s[i:])
        if not m:
            out.append(s[i:])
            return ''.join(out)
        a = i + m.end()
        out.append(s[i:a])
        e = s.index('\\end{' + m.group(1) + '}', a)
        body = s[a:e]
        c, t = (body.find('\\caption{'), body.find('\\begin{tabular'))
        if c > t >= 0:
            ce = _brace_end(body, c + len('\\caption'))
            cap, rest = (body[c:ce], body[:c] + body[ce:])
            k = rest.find('\\end{tabular}')
            lab = re.search('\\n?[ \\t]*\\\\label\\{[^}]*\\}', rest[k:]) if k >= 0 else None
            label = ''
            if lab:
                label = lab.group(0).strip()
                rest = rest[:k + lab.start()] + rest[k + lab.end():]
            rest = re.sub('\\n[ \\t]*\\n', '\n', rest)
            body = '  ' + cap + '\n' + ('  ' + label + '\n' if label else '') + rest.lstrip('\n')
        out.append(body)
        i = e


def solve():
    arms5 = rec('solve_arms_contrasts.json')
    c = arms5['contrasts']
    core = rec('lm_solve_paired.json')['paired_vs_init']['core_lm']
    pf = rec('solve_perframe_contrast.json')['D']['PF vs P']
    ww = rec('solve_window_weighted.json')
    assert (arms5['n_units'], arms5['n_videos'], arms5['gate_flagged_at_init']) == (44, 20, []), 'arms'
    assert [(c[k][m]['worse_gt10'], c[k][m]['better_gt10']) for k in ('C-B', 'C-D') for m in ('e_mm', 'offorbit_mm')] == [(9, 8), (8, 12), (15, 9), (7, 19)], 'arm tails'
    M = ('e_mm', 'offorbit_mm', 'gauge_mm')

    def cell(x):
        return (x['median_delta'], x['ci95_cluster'])
    arms = [('D $-$ A0', [cell(c['D-A0'][m]) for m in M]), ('B $-$ A0', [cell(c['B-A0'][m]) for m in M]), ('L $-$ A0', [cell(core[m]) for m in M]), ('C $-$ A0', [cell(c['C-A0'][m]) for m in M])]
    contr = [('C $-$ B', [cell(c['C-B'][m]) for m in M]), ('C $-$ D', [cell(c['C-D'][m]) for m in M])]
    rpf = (-pf['median_delta'], [-pf['median_delta_ci'][1], -pf['median_delta_ci'][0]])
    w = ww['window_weighted']
    got = [[(round(v, 1), [round(a, 1), round(b, 1)]) for v, (a, b) in r] for _, r in arms + contr]
    assert got == [[(-8.1, [-20.2, 0.3]), (-9.6, [-21.4, 13.1]), (-2.1, [-26.7, 5.2])], [(-2.6, [-15.0, 3.9]), (-8.7, [-16.8, 8.5]), (2.2, [-4.6, 6.2])], [(4.0, [-7.2, 19.8]), (-9.6, [-15.4, -0.9]), (1.9, [-15.0, 22.9])], [(-2.6, [-15.1, 7.9]), (-7.6, [-13.3, 8.2]), (0.1, [-15.4, 6.7])], [(-1.0, [-3.0, 2.0]), (-4.5, [-7.4, -0.8]), (0.0, [-6.9, 3.6])], [(3.1, [-2.4, 9.4]), (-8.2, [-15.2, -1.8]), (4.2, [0.7, 7.7])]], got
    assert (pf['n'], round(rpf[0], 1), [round(v, 1) for v in rpf[1]]) == (71, -10.5, [-16.1, -8.0]), 'per-frame'
    means = [round(c[k][m]['mean_delta'], 1) for k, m in (('C-B', 'e_mm'), ('C-B', 'offorbit_mm'), ('C-D', 'e_mm'), ('C-D', 'offorbit_mm'))]
    assert means == [7.2, 7.4, 10.0, 5.4], means
    assert (ww['n_eligible_windows'], ww['n_eligible_videos'], round(w['median_dM_mm'], 1), [round(v, 1) for v in w['dM_ci95']]) == (42, 18, -4.5, [-8.6, 0.0]), 'window-weighted'

    def f1(v):
        return f'{v:.1f}'.replace('-', '$-$')

    def cl(v, ci):
        return f1(v) + ' [' + f1(ci[0]) + ', ' + f1(ci[1]) + ']'
    lv = rec('solve_arms_levels.json')
    assert (lv['n_units'], lv['n_videos'], lv['corpus_counts']) == (44, 20, {'D': 42, 'H': 2}), lv['corpus_counts']
    L5, L4 = (lv['s524_level_median'], lv['s435_level'])
    lvl = {a: [L5[a][m] for m in M] for a in ('A0', 'B', 'C', 'D')}
    lvl['L'] = [L4['core_lm'][m] for m in M]
    for m_i, m in enumerate(M):
        assert abs(L4['init'][m] - lvl['A0'][m_i]) < 1e-09 and abs(L4['prod_final'][m] - lvl['C'][m_i]) < 1e-09, m
    got_l = {a: [round(v, 1) for v in x] for a, x in lvl.items()}
    assert got_l == {'A0': [89.7, 55.2, 63.4], 'B': [71.9, 55.5, 48.7], 'C': [86.0, 50.3, 61.1], 'D': [70.3, 56.1, 49.8], 'L': [108.0, 41.4, 86.2]}, got_l
    rot = {a: L4[k]['rot_deg'] for a, k in (('A0', 'init'), ('L', 'core_lm'), ('C', 'prod_final'))}
    rc = lv['s435_core_lm_rot_deg']
    assert ([round(rot[a], 1) for a in ('A0', 'L', 'C')], round(rc['median_delta'], 2), [round(v, 2) for v in rc['ci95_cluster']]) == ([11.1, 12.8, 12.8], -0.32, [-1.27, 0.05]), (rot, rc)
    tm = lv['s524_times']
    tms = (tm['A0']['median_s'], tm['B']['median_s'], tm['C_cpu']['median_s'], tm['C_cpu']['phases_median_s']['lm'], tm['D']['median_s'])
    assert [round(v) for v in tms] == [32, 32, 329, 296, 804], tms

    def g3(a):
        return ' / '.join((f'{v:.1f}' for v in lvl[a]))

    def cl(v, ci):
        txt = f1(v) + ' [' + f1(ci[0]) + ', ' + f1(ci[1]) + ']'
        return txt + '\\,$\\dagger$' if ci[0] > 0 or ci[1] < 0 else txt
    dA0 = {'A0': None, 'B': arms[1][1], 'L': arms[2][1], 'C': arms[3][1], 'D': arms[0][1]}
    labs = [('A0', 'Initialiser (A0)'), ('B', 'Initialiser + depth layers (B)'), ('L', 'Initialiser + LM solve (L)'), ('C', 'Ours: L + depth layers (C)'), ('D', 'Per-frame registration + re-anchoring, contact (D)')]
    lines = []
    bestm = [min((lvl[a][m_i] for a, _ in labs)) for m_i in range(3)]
    assert [round(v, 1) for v in bestm] == [70.3, 41.4, 48.7], bestm
    for a, lab in labs:
        cells = []
        for m_i in range(3):
            cells.append('--' if dA0[a] is None else cl(*dA0[a][m_i]))
            cells.append('\\textcolor{black!65}{' + f'{lvl[a][m_i]:.1f}' + '}')
        lines.append(f'    {lab} & ' + ' & '.join(cells) + ' \\\\')
    lines.append('    \\midrule')
    for lab, r in contr:
        lines.append(f'    {lab} (between arms, same windows) & ' + ' & & '.join((cl(v, ci) for v, ci in r)) + ' & \\\\')
    assert cl(*rpf) is not None
    assert lvl['B'][0] < lvl['C'][0] and lvl['D'][0] < lvl['C'][0], 'H1: B and D below C by arm median'
    cb = dict(contr)['C $-$ B']
    assert [round(cb[0][0], 2), [round(x, 2) for x in cb[0][1]], round(cb[1][0], 2), [round(x, 2) for x in cb[1][1]]] == [-0.96, [-2.99, 2.03], -4.46, [-7.37, -0.82]], cb
    body = '\\begin{table*}[t]\n  \\caption{\\textbf{Five pipeline variants (arms) on 44 previously used development windows (42 DexYCB, 2 HOI4D; 20 videos).} No arm lowered centre error measurably against A0, nor did ours (C) against B or D. Against D, C raised the median gauge error. Per window, centre error is a mean distance over frames, gauge and off-orbit errors are RMS; grey: unpaired medians, which can disagree with $\\Delta$ (B and C share $\\Delta=-2.6$ in centre error; both values are correct). Registered (the C, B and A0 contrasts), exploratory (development); the L row and the D rows are descriptive; no QC; D has no depth gate; D without re-anchoring and contact did not run. $\\Delta$: paired median difference (not additive), arm minus A0 or first minus second, 95\\% video-cluster CI; $\\dagger$: interval excludes zero (uncorrected). Means (centre, off-orbit): C $-$ B $+7.2$, $+7.4$\\,mm; C $-$ D $+10.0$, $+5.4$\\,mm. Windows worse/better by more than 10\\,mm (centre, off-orbit): C $-$ B 9/8, 8/12; C $-$ D 15/9, 7/19.}\n  \\label{tab:solve}\n  \\centering\\footnotesize\n  \\setlength{\\tabcolsep}{3.5pt}\n  \\begin{tabular}{@{}llrlrlr@{}}\n    \\toprule\n    & \\multicolumn{2}{c}{centre error (mm) $\\downarrow$} & \\multicolumn{2}{c}{off-orbit error (mm) $\\downarrow$} & \\multicolumn{2}{c}{gauge error (mm) $\\downarrow$} \\\\\n    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n    Arm & $\\Delta$ [95\\% CI] & median & $\\Delta$ [95\\% CI] & median & $\\Delta$ [95\\% CI] & median \\\\\n    \\midrule\n' + '\n'.join(lines) + '\n    \\bottomrule\n  \\end{tabular}\n  \\vspace{-2mm}\n\\end{table*}\n'
    write('solve.tex', body)


def optim():
    so = {r['arm']: r for r in rec('summary_second_order.json')['table']}
    fo = {r['arm']: r for r in rec('summary_first_order.json')['table']}
    ms = {r['arm']: r for r in rec('summary_multistart.json')['table']}
    names = {'gn': 'Gauss--Newton$^{\\dagger}$', 'gn_ls': 'GN, line search', 'dogleg': 'Dogleg', 'gd': 'Gradient descent', 'gd_jacobi': 'Jacobi descent', 'adam': 'Adam', 'lbfgs': 'L-BFGS', 'ms': 'Multi-start LM'}
    lines = []
    for grp, tab in ((('gn', 'gn_ls', 'dogleg'), so), (('gd', 'gd_jacobi', 'adam', 'lbfgs'), fo), (('ms',), ms)):
        for a in grp:
            r = tab[a]
            pc = r['merit_ppm_vs_lm_median'] / 10000.0
            ppm = f'{pc:+.1f}' if abs(pc) >= 1 else f'{pc:+.3f}'
            de, lo, hi = (-r['de_mm_median'] + 0.0, -r['de_mm_ci'][1] + 0.0, -r['de_mm_ci'][0] + 0.0)
            units = f'{r['n']} ({r['n'] - r['fail']})' if r['fail'] else f'{r['n']}'
            lines.append(f'    {names[a]} & {units} & {r['fail']} & ${ppm}$ & ${de:+.1f}$ $[{lo:.1f}, {hi:.1f}]$ & {r['sec_ratio_median']:.2f} \\\\')
        lines.append('    \\midrule')
    lines.pop()
    body = '\\begin{table}[h]\n  \\centering\\footnotesize\n  \\setlength{\\tabcolsep}{1.5pt}\n  \\begin{tabular}{@{}lrrrrr@{}}\n    \\toprule\n    Step rule & Units & Fail & $E$ vs LM (\\%) & $\\Delta$err (mm), 95\\% CI & Time \\\\\n    \\midrule\n' + '\n'.join(lines) + "\n    \\bottomrule\n  \\end{tabular}\n  \\caption{\\textbf{No step rule lowered the centre error with an interval that excludes zero, and on the eight tested development units no first-order method reached LM's objective within the tested budgets.} Step rules applied to the same objective (development, CPU, descriptive). Entries are medians of paired differences. $E$ vs LM compares the step rule's objective with LM's, re-evaluated at one fine-level refresh. $\\Delta$err is LM minus the step rule, so negative values favour LM, as in \\cref{tab:solve} and \\cref{tab:ablate}; brackets give a bootstrap 95\\% interval of the median. Time is a ratio to LM in the same process. For multi-start, $E$ is compared with its initial start. $^{\\dagger}$Gauss--Newton failed on 2 of 44 units; its medians cover the 42 where it did not fail (with both failures scored, $-1.26$\\,mm).}\n  \\label{tab:optim}\n\\end{table}\n"
    write('optim.tex', body)


def stride():
    d = rec('stride_safe_carry.json')
    tol = d['tolerances']
    rows = []
    for k, name, why in (('D', 'DexYCB', 'annotation jitter'), ('H', 'HOI4D', 'annotation jitter'), ('O', 'OakInk2 MoCap', 'motion (hold)')):
        g = d['domains'][k]
        pr = g['safe']['primary']
        scored = g['domain_safe_channels']
        cell = lambda ch: str(pr[ch]['ub']) if ch in scored else '--'
        rows.append(f'    {name} ({g['rate_hz']:.0f}\\,fps) & {cell('O-c')} & {cell('H-j')} & {cell('O-R')} & {why} \\\\')
    assert rows[2].split('&')[1:4] == [' 5 ', ' 3 ', ' 1 '], rows[2]
    body = '\\begin{table}[h]\n  \\centering\\footnotesize\n  \\setlength{\\tabcolsep}{3pt}\n  \\begin{tabular}{@{}lcccl@{}}\n    \\toprule\n    & \\multicolumn{3}{c}{safe stride (GT frames)} & \\\\\n    \\cmidrule(lr){2-4}\n    Domain (GT rate) & centre & hand & rotation & consistent with \\\\\n    \\midrule\n' + '\n'.join(rows) + "\n    \\bottomrule\n  \\end{tabular}\n  \\caption{\\textbf{Spline carry of ground-truth motion} (registered): the largest stride whose carry, at that and every smaller stride, stays within the registered tolerances (centre 2.5\\,mm, hand joints 2.5\\,mm, rotation 2.9$^\\circ$), applied in each contact phase to the upper end of a 95\\% sequence-cluster bootstrap interval (2,000 resamples) for the 95th percentile of per-frame error over non-key frames. A phase counts if at least 10 sequences contribute non-key frames to it at stride 6: all four phases for HOI4D and OakInk2, but only approach, onset and hold for DexYCB, whose 800 sequences hold 3 release events. HOI4D object poses are interpolated between 1.5\\,Hz knots and are not scored. Consistent with: the source of the carry error that the ground truth's midpoint residual and the slope of error against stride point to (evidence, not a test; \\cref{sec:supp-ablate}).}\n  \\label{tab:stride}\n\\end{table}\n"
    write('stride.tex', body)


def hyper():
    rows = [('Windows', 'internet video: 10\\,s, consecutive, no overlap; stride $\\nu_v=\\max(1,\\lfloor f/10+\\tfrac12\\rfloor)$. Held-out units instead: HOI4D windows of 150 frames at 15\\,fps (stride 2) and whole DexYCB clips of 68--74 frames at 30\\,fps (stride 3)'), ('Prompt and keyframe', "hand--object detector on every 5th processed frame (first and last always); a detection needs hand and object scores $\\ge0.9$ and the object-contact class; the highest interaction score sets the keyframe; mask area 0.2--40\\% of the image and fewer than 3 image borders touched, else the window is refused; SAM~2.1 is prompted once, on the keyframe, and propagates forward and backward. Benchmark scripts on DexYCB and HOI4D instead take processed frame 0 with the dataset's object label"), ('SAM~3D', 'once per window, from the keyframe; structured-latent flow integrated with 12 second-order Adams--Bashforth steps'), ('Metric scale and intrinsics', "per frame, the mean HaWoR joint depth of the right hand over the mean MoGe-2 depth at those joints' pixels; the window's factor is the median over frames; right hand only, so a window with no valid right-hand joint gets no finite scale and is refused (an opt-in \\texttt{auto} setting falls back to the left hand); estimated focal lengths scaled by 1.104"), ('Tracks', '96 CoTracker3 queries drawn uniformly in the keyframe mask (seed 0), tracked forward and over the reversed prefix'), ('Trimmed ICP', '120 rotation hypotheses; centroid started 0.25 diameters behind the median point; 50 iterations on 512 points and 1024 samples; correspondences beyond 2.5$\\times$ the median residual dropped; score: Chamfer distance truncated at 0.1 diameter plus 0.1 diameter times the outside-mask fraction; rigid, with initial scale from the 5--95\\% extent over $0.9D$'), ('EPnP-RANSAC', "inlier threshold 3.0\\,px, 200 iterations, then LM refinement; confidence: OpenCV default, not recorded; a frame with fewer than 6 visible tracks or inliers, or failing the distance test of \\cref{alg:window}, takes the nearest accepted frame's rotation, placed as \\cref{alg:window} states"), ('Transport', 'log-domain Sinkhorn, float32; up to 1024 points and 2048 mesh samples; $\\varepsilon$ 0.05, $2\\times10^{-3}$, $10^{-4}$ and KL penalties (points, mesh) 10 and 1, 1 and 0.1, 0.1 and $10^{-3}$ at the three levels, both in units of $(s_0D)^2$; 50 iterations cold and 10 warm, no tolerance; mesh mass: visible samples (z-buffer, hand excluded), uniform at the first level, weighted by $\\cos\\theta/z^2$ after; nearest 32 samples at the last level'), ('LM', "levels are the three transport levels; $\\mu_0$ $10^{-2}$, $3\\times10^{-3}$, $10^{-3}$; $\\mu\\times5$ on a rejected step, $\\div3$ on an accepted one; 10, 5 and 20 iterations; stop when the relative decrease is below $10^{-4}$ or $\\mu>10^6$; ridge $10^{-9}$; a step is rejected if the clip's median object depth falls to 0.05\\,m; Cauchy scale 4\\,px; track weight $1/(s_0D)^2$, a fixed choice, not a unit normalisation (track residuals in px, pointmap residuals in m), so the two terms' balance varies with object size; prior weights $\\omega_s\\bar N$ and $\\omega_v\\bar N$ on the squared residuals, with $\\omega_s=0.1$ and $\\omega_v=1$"), ('Depth check and anchor', "$\\kappa=|\\med_kd_k|\\ge0.25$ over frames with at least 10 valid points (no check below 3 such frames); re-solve from the same start with weight 500 on $r_k^2$, $r_k=\\mathrm{sign}(d_k)\\max(0,|d_k|-0.2)$, on the template centroid's log depth, with $\\partial r_k/\\partial\\sigma$ set to 0 (the anchor supplies no scale derivative; the scale can still change through its coupling with the other terms in the joint solve)"), ('Re-anchor', "flagged clip: every frame with $|d_k|>0.2$ or $q_k>1$ (projected-centroid offset from the mask centroid over $\\sqrt{\\text{mask area}/\\pi}$) moves, in translation, to its observed depth on the mask-centroid ray; unflagged clip: only frames with fewer than 5 visible tracks, and only on third-person video; such a frame outside the depth band slides along its own ray to the band edge, and such a frame off its mask moves onto the mask-centroid ray at its depth (after any slide); safety limits on every move: log-depth shift at most 0.3, and a frame whose mask area is under 0.25 of the clip's median never moves away from the camera"), ('Contact', 'third-person video only, enabled per run (off in the root configuration); shifts the object along the camera ray by the posterior mean over slides of $-300$ to 300\\,mm in 2\\,mm steps, whose likelihood depends on the mean distance from the hand joints to the object surface; geometric ruler (prior s.d.\\ $\\sigma_c$ 86.4\\,mm, likelihood width $\\tau$ 20\\,mm); the 56\\,mm touch gate is off'), ('Spline carry', 'object poses by a $C^2$ natural cubic spline through every solved processed frame, and wrist poses through the processed frames with a valid hand (rotations as sign-continuous, renormalised unit quaternions; no extrapolation: hand frames past the first or last valid processed frame are held and masked); hand vertices linear in the wrist frame; a carried frame is valid only if both bracketing knots are'), ('QC', 'logistic score on five features and an intercept (reprojection IoU, share of frames passing a Laplacian check at 0.01, transport residual, penetration depth, track inlier ratio); refuse if $1-P(\\text{good})$ exceeds a threshold set by split conformal calibration on good clips ($\\alpha=0.1$); same rule in every domain. Threshold 0.6908, six logistic weights and the feature standardisation in the rule file the held-out run loaded, outside the pinned tree, whose hash the protocol registered')]
    body = "\\begin{table*}[t]\n  \\caption{\\textbf{Settings of our pipeline} (production defaults at the tree pinned for our side of the held-out test; a study may override them, and each study's record names its overrides). $s_0D$: initial scale times $D$, the bounding-box diagonal of the mesh's surface samples, at most 1 since the mesh is normalised to a unit diagonal; $\\bar N$: mean valid points per frame; $d_k$, $\\kappa$: main \\cref{sec:layers}. ``Not recorded'': the pinned tree does not hold the value.}\n  \\label{tab:hyper}\n  \\centering\\footnotesize\n  \\setlength{\\tabcolsep}{3pt}\n  \\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.13\\linewidth}>{\\raggedright\\arraybackslash}p{0.83\\linewidth}@{}}\n    \\toprule\n    Setting & Value \\\\\n    \\midrule\n" + '\n'.join((f'    {a} & {b} \\\\' for a, b in rows)) + '\n    \\bottomrule\n  \\end{tabular}\n\\end{table*}\n'
    write('hyper.tex', body)


def weak_rows():
    d = rec('weak_rows_summary.json')
    assert (d['n_windows'], d['n_videos'], d['rows_total'], d['rows_scored'], d['nboot']) == (44, 20, 1164, 1164, 10000), 'weak rows'
    C = d['cells']
    strata = [('H1/abs', 'fewer than 20 EPnP inliers'), ('H1/median', 'EPnP inliers, median split'), ('H2/abs', 'within one frame of a track or mask gap'), ('H2/median', 'gap distance, median split'), ('H3/abs', 'mask area below 2{,}500\\,px'), ('H3/median', 'mask area, median split')]

    def g(v, ci, n=None):
        t = f'{v:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]'.replace('-', '$-$')
        star = '$^{\\ast}$' if ci[0] > 0 or ci[1] < 0 else ''
        return t + star + (f' ({n})' if n is not None else '')
    lines, verdicts = ([], [])
    for met, mlab in (('centre', 'Centre error'), ('offorbit', 'Off-orbit error')):
        lines.append('    \\multicolumn{4}{@{}l}{\\emph{' + mlab + '}, C $-$ B per row, mm} \\\\')
        for key, lab in strata:
            c = C[f'C-B|{met}|{key}']
            verdicts.append(c['verdict'])
            lines.append(f'    {lab} & ' + g(c['low']['median'], c['low']['ci95'], c['low']['n_windows']) + ' & ' + g(c['high']['median'], c['high']['ci95']) + ' & ' + g(c['diff']['median'], c['diff']['ci95']) + ' \\\\')
        a = C[f'C-B|{met}|H1/abs']['all']
        lines.append('    all rows & ' + g(a['median'], a['ci95'], a['n_windows']) + ' & & \\\\')
    assert verdicts == ['NOT SUPPORTED'] * 12, verdicts
    chk = (C['C-B|centre|H1/abs']['low'], C['C-B|centre|H2/abs']['low'], C['C-B|centre|H3/abs']['low'], C['C-B|offorbit|H1/abs']['high'], C['C-B|offorbit|H2/median']['diff'])
    assert [(round(x['median'], 2), [round(v, 2) for v in x['ci95']]) for x in chk] == [(1.43, [-1.52, 12.75]), (9.92, [-8.77, 31.92]), (-1.24, [-4.1, 6.04]), (-2.1, [-4.3, -0.48]), (5.71, [0.87, 14.03])], chk
    body = "\\begin{table*}[t]\n  \\caption{\\textbf{In none of the three weak strata of rows (few EPnP inliers, near a gap, small mask) did the per-row change against the initialiser with depth layers (C $-$ B) have an interval below zero (0 of 12 cells under the claim rule fixed before the analysis).} A fourth stratum, rows that the initialiser's gate rejected, was descriptive and is not shown. Here C $-$ B is the solve together with its interaction with the depth layers, computed per row, a different statistic from the per-window C $-$ B of \\cref{tab:solve}. Post hoc (designed after the results of \\cref{tab:solve} were known; strata and claim rule fixed in writing before this analysis ran), exploratory, on the 44 spent development windows of \\cref{tab:solve} (20 videos, 1{,}164 rows). Each cell: per-row paired difference, median within window and stratum, then median over windows, with a 95\\% video-cluster bootstrap CI; low: the weak stratum named (windows in brackets); high: the complementary eligible rows (rows tied at a median cut fall in neither stratum, and the gap-distance median split also omits rows of windows with no gap); for gap distance, the median split excludes 703 rows in 27 gap-free windows and 43 median-tied rows; low $-$ high: median over windows that hold both. Negative is better. $^{\\ast}$Interval excludes 0. In off-orbit error the well-tracked stratum (at least 20 inliers) had an interval below zero ($-2.10$\\,mm [$-4.30$, $-0.48$]); the direct difference between strata was not resolved (+2.85 [$-4.54$, +8.56]). That interval and the all-rows one cross zero once each window's off-orbit error is divided by its $\\hat\\lambda$ (post hoc).}\n  \\label{tab:s566}\n  \\centering\\footnotesize\n  \\setlength{\\tabcolsep}{5pt}\n  \\begin{tabular}{@{}llll@{}}\n    \\toprule\n    Weak stratum & low & high & low $-$ high \\\\\n    \\midrule\n" + '\n'.join(lines) + '\n    \\bottomrule\n  \\end{tabular}\n\\end{table*}\n'
    write('weak_rows.tex', body)

def rec(name):
    return json.loads((REC / name).read_text())


def write(name, body):
    """Write `<OUT>/<name>`: a two-line header, then the table with its caption above it."""
    head = ("% GENERATED by reproduce/make_released_tables.py from reproduce/records/.\n"
            "% Derived from the paper source's tools/make_tables.py; see that file's docstring.\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(head + caption_above(body))


def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="directory for the regenerated tables")
    OUT = ap.parse_args(argv).out
    for fn in (solve, optim, stride, hyper, weak_rows):
        fn()
    print("tables written:", sorted(p.name for p in OUT.glob("*.tex")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
