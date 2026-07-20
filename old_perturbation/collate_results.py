"""
collate_results.py
==================
Reads existing results/ TSVs and perturbed_inputs/ files to produce
CSV output — no model re-running required.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

STRAINS = {
    "10465": "all_embeded_genomes/GCF_000010465.1_ASM1046v1__254__2007__0000.txt",
    "12045": "all_embeded_genomes/GCF_000012045.1_ASM1204v1__250__2005__0000.txt",
    "13425": "all_embeded_genomes/GCF_000013425.1_ASM1342v1__8__2006__0000.txt",
    "13465": "all_embeded_genomes/GCF_000013465.1_ASM1346v1__8__2006__0000.txt",
}

GENE_NAMES_FILE = "gene_names.txt"
RESULTS_DIR     = Path("results")
PERTURBED_DIR   = Path("perturbed_inputs")
VARIATIONS      = [1, 10, 100]

OUTPUT_ALL      = "perturbation_all_results.csv"
OUTPUT_FLIPPED  = "perturbation_flipped_summary.csv"
OUTPUT_STRATEGY = "perturbation_strategy_summary.csv"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_gene_categories(path=GENE_NAMES_FILE):
    def categorize(name):
        n = name.lower()
        if any(x in n for x in ['phage','phi ','hk97','capsid','terminase','portal',
                                  'tail','endolysin','integrase','xkd','head-tail',
                                  'lysostaphin','autolysin','tape measure','mu-f']):
            return 'phage'
        elif any(x in n for x in ['ribosom','rna pol','dna pol','dnaa','gyrase',
                                    'topoisomerase','recf','dnab','fts','mur',
                                    '30s','50s','trna','rrna','atp synthase','mnma']):
            return 'core'
        elif any(x in n for x in ['tet(','bcrb','bacitracin','meca','efflux','stk1',
                                    'iscr','msr','macrolide','beta-lacta','blaz',
                                    'blar','blai','erm','type i restriction']):
            return 'resistance'
        elif any(x in n for x in ['iron','isca','iscu','fe-s','fur','iron-regulated']):
            return 'iron_fes'
        else:
            return 'other'
    lookup = {}
    with open(path) as f:
        for idx, line in enumerate(f):
            name = line.strip()
            if name:
                lookup[idx] = (name, categorize(name))
    return lookup

def read_genome(path):
    genes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            genes.append([int(parts[0]), int(parts[1])])
    return genes

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading gene categories...")
    gene_cats = load_gene_categories()

    print("Loading original genomes...")
    originals = {sid: read_genome(path) for sid, path in STRAINS.items()
                 if Path(path).exists()}

    # Parse result filenames: {strain}_{strategy}_n{flips}_s{seed}_var{variation}.tsv
    pattern = re.compile(
        r'^(?P<strain>\d+)_(?P<strategy>.+)_n(?P<flips>\d+)_s(?P<seed>\d+)_var(?P<variation>\d+)\.tsv$'
    )

    all_records = []
    tsv_files = sorted(RESULTS_DIR.glob("*.tsv"))
    print(f"Found {len(tsv_files)} result TSVs...")

    for tsv_path in tsv_files:
        m = pattern.match(tsv_path.name)
        if not m:
            continue

        strain_id  = m.group('strain')
        strategy   = m.group('strategy')
        n_flips    = int(m.group('flips'))
        seed       = int(m.group('seed'))
        variation  = int(m.group('variation'))

        baseline_genes = originals.get(strain_id)
        if baseline_genes is None:
            continue
        n_genes = len(baseline_genes)

        # Load perturbed genome to find which genes were flipped
        perturbed_path = PERTURBED_DIR / f"{strain_id}_{strategy}_n{n_flips}_s{seed}.txt"
        if not perturbed_path.exists():
            continue
        perturbed_genes = read_genome(perturbed_path)
        flipped_set = {i for i, (orig, pert) in enumerate(zip(baseline_genes, perturbed_genes))
                       if orig[0] != pert[0]}

        df_out = pd.read_csv(tsv_path, sep='\t')[['gene_idx', 'logit_prob']]

        for _, row in df_out.iterrows():
            idx = int(row['gene_idx'])
            gene_name, cat = gene_cats.get(idx, ('unknown', 'unknown'))
            orig = baseline_genes[idx][0] if idx < n_genes else None
            new  = perturbed_genes[idx][0] if idx < n_genes else None
            all_records.append({
                'strain':        strain_id,
                'strategy':      strategy,
                'flip_size':     n_flips,
                'seed':          seed,
                'variation':     variation,
                'gene_idx':      idx,
                'gene_name':     gene_name,
                'category':      cat,
                'was_flipped':   idx in flipped_set,
                'orig_presence': orig,
                'new_presence':  new,
                'logit_prob':    row['logit_prob'],
            })

    if not all_records:
        print("No records found — check results/ and perturbed_inputs/ exist.")
        return

    print(f"Collated {len(all_records):,} records. Computing stats...")
    df = pd.DataFrame(all_records)

    def add_z(group):
        mu, sd = group['logit_prob'].mean(), group['logit_prob'].std()
        group = group.copy()
        group['z_score'] = (group['logit_prob'] - mu) / sd if sd > 0 else 0.0
        return group

    df = df.groupby(['strain','variation','orig_presence'], group_keys=False).apply(add_z).reset_index(drop=True)

    df['suspect'] = (
        ((df['orig_presence'] == 0) & (df['logit_prob'] > 0)) |
        ((df['orig_presence'] == 1) & (df['logit_prob'] < 0))
    )

    flipped_df = df[df['was_flipped']].copy()

    pivot = flipped_df.pivot_table(
        index=['strain','strategy','flip_size','seed','gene_idx','gene_name',
               'category','orig_presence','new_presence'],
        columns='variation',
        values=['logit_prob','z_score','suspect'],
        aggfunc='first'
    )
    pivot.columns = [f'{col[0]}_var{col[1]}' for col in pivot.columns]
    pivot = pivot.reset_index()

    for v in VARIATIONS:
        if f'suspect_var{v}' in pivot.columns:
            pivot[f'suspect_var{v}'] = pivot[f'suspect_var{v}'].fillna(False)

    if all(f'suspect_var{v}' in pivot.columns for v in VARIATIONS):
        pivot['consistent_across_variations'] = (
            pivot[['suspect_var1','suspect_var10','suspect_var100']].nunique(axis=1) == 1
        )
        pivot['only_suspect_at_100'] = (
            ~pivot['suspect_var1'] & ~pivot['suspect_var10'] & pivot['suspect_var100']
        )
        pivot['suspect_at_all'] = (
            pivot['suspect_var1'] & pivot['suspect_var10'] & pivot['suspect_var100']
        )

    summary = None
    if not pivot.empty and 'consistent_across_variations' in pivot.columns:
        summary = pivot.groupby(['strategy','flip_size','category']).agg(
            n_genes=('gene_idx','count'),
            pct_consistent=('consistent_across_variations','mean'),
            pct_suspect_all=('suspect_at_all','mean'),
            pct_only_at_100=('only_suspect_at_100','mean'),
        ).reset_index()

    print(f"Writing CSVs...")
    df.to_csv(OUTPUT_ALL, index=False)
    pivot.to_csv(OUTPUT_FLIPPED, index=False)
    if summary is not None:
        summary.to_csv(OUTPUT_STRATEGY, index=False)

    print(f"Done.")
    print(f"  {OUTPUT_ALL}          — {len(df):,} rows")
    print(f"  {OUTPUT_FLIPPED}  — {len(pivot):,} rows")
    if summary is not None:
        print(f"  {OUTPUT_STRATEGY} — {len(summary):,} rows")


if __name__ == '__main__':
    main()
