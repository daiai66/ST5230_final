import json

path = r'D:\aaastats\5230\result\topic 3.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

new_source = [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import re\n",
    "from scipy import stats\n",
    "\n",
    "# --- Extract Post.Mean and Post.SD directly from summary table ---\n",
    "# params has extra random-effect predictions beyond fep+vcp; slice to match summary\n",
    "names = result.model.fep_names + result.model.vcp_names\n",
    "n_params = len(names)\n",
    "means = result.params[:n_params]\n",
    "\n",
    "# Parse Post.SD from summary (robust across statsmodels versions)\n",
    "_sds = []\n",
    "for line in result.summary().as_text().split('\\n'):\n",
    "    m = re.search(r'\\s+[MV]\\s+([-\\d.]+)\\s+([\\d.]+)', line)\n",
    "    if m:\n",
    "        _sds.append(float(m.group(2)))\n",
    "sds = np.array(_sds)\n",
    "assert len(sds) == len(means), f'SD parse mismatch: {len(sds)} vs {len(means)}'\n",
    "\n",
    "\n",
    "def clean_name(n):\n",
    "    n = n.replace(\"C(responsibility, Treatment('neutral'))\", 'resp')\n",
    "    n = n.replace(\"C(role, Treatment('role_A'))\", 'role')\n",
    "    n = n.replace('[T.', '[')\n",
    "    return n\n",
    "\n",
    "clean_names = [clean_name(n) for n in names]\n",
    "fixed_idx       = list(range(len(result.model.fep_names)))\n",
    "variance_idx    = list(range(len(result.model.fep_names), len(names)))\n",
    "intercept_idx   = [i for i in fixed_idx if 'resp' not in clean_names[i] and 'role' not in clean_names[i]]\n",
    "resp_main_idx   = [i for i in fixed_idx if 'resp[' in clean_names[i] and ':' not in clean_names[i]]\n",
    "role_main_idx   = [i for i in fixed_idx if 'role[' in clean_names[i] and ':' not in clean_names[i]]\n",
    "interaction_idx = [i for i in fixed_idx if ':' in clean_names[i]]\n",
    "\n",
    "groups = [\n",
    "    ('Intercept',                intercept_idx,   '#5b7fba'),\n",
    "    ('Responsibility (main)',    resp_main_idx,   '#e07b3a'),\n",
    "    ('Role (main)',              role_main_idx,   '#57a86b'),\n",
    "    ('Resp x Role (interaction)', interaction_idx, '#9b59b6'),\n",
    "    ('Variance (log SD)',        variance_idx,    '#c0392b'),\n",
    "]\n",
    "plot_groups = [(lbl, idx, col) for lbl, idx, col in groups if len(idx) > 0]\n",
    "\n",
    "fig = plt.figure(figsize=(22, 10))\n",
    "fig.patch.set_facecolor('white')\n",
    "fig.suptitle('Posterior Distributions - BinomialBayesMixedGLM (fit_map)\\n'\n",
    "             'Curve = N(Post.Mean, Post.SD) | line = mean | bar = 95% CI | red dashed = 0',\n",
    "             fontsize=12, fontweight='bold')\n",
    "\n",
    "width_ratios = [max(1, len(g[1])) for g in plot_groups]\n",
    "axes_list = fig.subplots(1, len(plot_groups),\n",
    "                         gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.05})\n",
    "if len(plot_groups) == 1:\n",
    "    axes_list = [axes_list]\n",
    "\n",
    "for ax, (group_label, idxs, color) in zip(axes_list, plot_groups):\n",
    "    ax.set_facecolor('white')\n",
    "    all_vals = [v for idx in idxs for v in [means[idx]-3.5*sds[idx], means[idx]+3.5*sds[idx]]]\n",
    "    y_min, y_max = min(all_vals), max(all_vals)\n",
    "    pad = (y_max - y_min) * 0.10\n",
    "    ax.set_ylim(y_min - pad, y_max + pad)\n",
    "\n",
    "    for i, idx in enumerate(idxs):\n",
    "        mu, sig = means[idx], sds[idx]\n",
    "        x = np.linspace(mu - 3.8*sig, mu + 3.8*sig, 300)\n",
    "        y_pdf = stats.norm.pdf(x, mu, sig)\n",
    "        y_norm = y_pdf / y_pdf.max() * 0.80\n",
    "        ax.fill_betweenx(x, i, i + y_norm, alpha=0.50, color=color)\n",
    "        ax.plot(i + y_norm, x, color=color, lw=1.2)\n",
    "        ci_lo, ci_hi = mu - 1.96*sig, mu + 1.96*sig\n",
    "        ax.plot([i+0.05, i+0.75], [mu, mu], color='#111111', lw=1.8, zorder=4)\n",
    "        ax.plot([i+0.15, i+0.15], [ci_lo, ci_hi], color='#555555', lw=1.2, zorder=3)\n",
    "        if idx in fixed_idx:\n",
    "            ax.axhline(0, color='red', lw=0.7, linestyle='--', alpha=0.45)\n",
    "        lbl = clean_names[idx]\n",
    "        if ':' in lbl:\n",
    "            p = lbl.split(':')\n",
    "            lbl = p[0].split('[')[-1].rstrip(']') + 'x' + p[1].split('[')[-1].rstrip(']')\n",
    "        elif '[' in lbl:\n",
    "            lbl = lbl.split('[')[-1].rstrip(']')\n",
    "        ax.text(i+0.40, y_min - pad*0.5, lbl, ha='center', va='top', fontsize=7.5, color='#333333')\n",
    "\n",
    "    ax.set_xticks([])\n",
    "    ax.set_title(group_label, fontsize=10, fontweight='bold', color=color, pad=8)\n",
    "    ax.spines[['top','right','bottom']].set_visible(False)\n",
    "    ax.spines['left'].set_color('#cccccc')\n",
    "    ax.yaxis.grid(True, color='#eeeeee', linewidth=0.5)\n",
    "    ax.set_axisbelow(True)\n",
    "    ax.set_xlim(-0.1, len(idxs))\n",
    "\n",
    "axes_list[0].set_ylabel('Parameter Value (log-odds)', fontsize=10)\n",
    "plt.tight_layout()\n",
    "plt.savefig('posterior_distributions.png', dpi=200, bbox_inches='tight', facecolor='white')\n",
    "plt.show()\n",
    "print('Saved -> posterior_distributions.png')\n"
]

for cell in nb['cells']:
    if cell.get('id') == '7751a613':
        cell['source'] = new_source
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Cell 7751a613 updated.')
        break
else:
    print('Cell not found!')

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Done.')
