#!/usr/bin/env python3
"""
clean_gwas.py

Standardises GWAS summary statistics columns, filters to P < 5e-8,
removes ambiguous SNPs (A/T, C/G), and removes effect-size (BETA) outliers.
Maps variant IDs to the 1000 Genomes reference panel BIM IDs for seamless PLINK clumping.
Handles Xiong 2025 traits and is ready for Du 2025 traits when downloaded.
"""

import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np

# Nucleotide complements for strand-flip checks
COMPLEMENT = {
    'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
    'a': 't', 't': 'a', 'c': 'g', 'g': 'c'
}

def get_complement(allele):
    return "".join(COMPLEMENT.get(base, base) for base in allele)

def is_ambiguous(a1, a2):
    """Check if alleles form an ambiguous A/T or C/G pair."""
    a1, a2 = a1.upper(), a2.upper()
    return (a1 == 'A' and a2 == 'T') or (a1 == 'T' and a2 == 'A') or \
           (a1 == 'C' and a2 == 'G') or (a1 == 'G' and a2 == 'C')

def match_alleles(g_a1, g_a2, r_a1, r_a2):
    """
    Check if GWAS alleles (g_a1, g_a2) match reference alleles (r_a1, r_a2)
    under exact, swapped, or strand-flipped configurations.
    """
    g_a1, g_a2 = g_a1.upper(), g_a2.upper()
    r_a1, r_a2 = r_a1.upper(), r_a2.upper()
    
    # Exact match
    if g_a1 == r_a1 and g_a2 == r_a2:
        return True, False
    # Swapped alleles
    if g_a1 == r_a2 and g_a2 == r_a1:
        return True, True
    
    # Strand flipped exact
    g_a1_comp, g_a2_comp = get_complement(g_a1), get_complement(g_a2)
    if g_a1_comp == r_a1 and g_a2_comp == r_a2:
        return True, False
    # Strand flipped swapped
    if g_a1_comp == r_a2 and g_a2_comp == r_a1:
        return True, True
        
    return False, False

def load_bim_lookup(bim_path, candidate_coords):
    """
    Load matching variant IDs from the reference BIM file for candidate (CHR, BP) coordinates.
    This saves memory by ignoring non-significant variants.
    """
    if not os.path.exists(bim_path):
        print(f"[WARN] Reference BIM file not found at '{bim_path}'. Variant mapping will fall back to coordinate strings.")
        return {}
        
    print(f"[INFO] Building lookup from reference BIM: '{bim_path}'...")
    lookup = {}
    
    # Read BIM file line-by-line to avoid loading 6M+ rows into memory fully
    match_count = 0
    with open(bim_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            # BIM format: CHR, ID, DIST, BP, A1, A2
            try:
                chr_num = int(parts[0])
                bp_pos = int(parts[3])
            except ValueError:
                # Handle non-integer chromosomes if any
                chr_num = parts[0]
                bp_pos = int(parts[3])
                
            coord = (chr_num, bp_pos)
            if coord in candidate_coords:
                var_id = parts[1]
                a1 = parts[4]
                a2 = parts[5]
                if coord not in lookup:
                    lookup[coord] = []
                lookup[coord].append((var_id, a1, a2))
                match_count += 1
                
    print(f"[INFO] Loaded {match_count} reference variants matching candidate coordinates.")
    return lookup

def clean_gwas_df(df, bim_lookup, p_thresh, z_thresh, trait_name):
    """
    Filter and clean summary statistics DataFrame:
    1. Filter P-value.
    2. Remove ambiguous SNPs.
    3. Remove outliers on BETA.
    4. Map variant IDs to reference panel IDs.
    """
    total_raw = len(df)
    
    # 1. P-value filter
    df = df[df['P'] < p_thresh].copy()
    total_sig = len(df)
    if total_sig == 0:
        print(f"  No variants passed P < {p_thresh} for {trait_name}.")
        return pd.DataFrame()
        
    # 2. Ambiguous SNPs filter
    ambig_mask = df.apply(lambda row: is_ambiguous(row['A1'], row['A2']), axis=1)
    df_clean = df[~ambig_mask].copy()
    n_ambig = sum(ambig_mask)
    
    # 3. Outlier SNPs filter (Z-score on BETA)
    n_outliers = 0
    if len(df_clean) >= 3:
        mean_beta = df_clean['BETA'].mean()
        std_beta = df_clean['BETA'].std()
        if std_beta > 0:
            z_scores = (df_clean['BETA'] - mean_beta).abs() / std_beta
            outlier_mask = z_scores > z_thresh
            n_outliers = sum(outlier_mask)
            if n_outliers > 0:
                print(f"  [OUTLIER] Removing {n_outliers} SNPs with BETA Z-score > {z_thresh} in {trait_name}:")
                outliers = df_clean[outlier_mask]
                for idx, row in outliers.iterrows():
                    print(f"    - SNP: {row['SNP']} (rsid: {row['rsid']}), BETA: {row['BETA']:.4f}, Z: {z_scores[idx]:.2f}, P: {row['P']:.2e}")
                df_clean = df_clean[~outlier_mask].copy()
                
    # 4. Map Variant IDs to Reference Panel IDs
    ref_ids = []
    unmapped_count = 0
    
    for _, row in df_clean.iterrows():
        coord = (int(row['CHR']), int(row['BP']))
        g_a1, g_a2 = row['A1'], row['A2']
        
        mapped_id = None
        if coord in bim_lookup:
            # Check for allele compatibility
            for ref_id, r_a1, r_a2 in bim_lookup[coord]:
                matched, swapped = match_alleles(g_a1, g_a2, r_a1, r_a2)
                if matched:
                    mapped_id = ref_id
                    break
                    
        if mapped_id:
            ref_ids.append(mapped_id)
        else:
            # Fall back to constructed coordinate ID if reference mapping is not found
            fallback_id = f"{row['CHR']}:{row['BP']}:{g_a1}:{g_a2}"
            ref_ids.append(fallback_id)
            unmapped_count += 1
            
    df_clean['SNP'] = ref_ids
    
    print(f"  Trait: {trait_name}")
    print(f"    Significant (P < {p_thresh}) : {total_sig} / {total_raw} SNPs")
    print(f"    Removed Ambiguous (A/T, C/G)  : {n_ambig} SNPs")
    print(f"    Removed Outliers (Z > {z_thresh})  : {n_outliers} SNPs")
    print(f"    Mapped to Reference IDs       : {len(df_clean) - unmapped_count} SNPs")
    if unmapped_count > 0:
        print(f"    Unmapped (fallback IDs used)  : {unmapped_count} SNPs")
        
    # Order and standardise columns
    cols = ['CHR', 'BP', 'SNP', 'rsid', 'A1', 'A2', 'BETA', 'P']
    return df_clean[cols]

def main():
    parser = argparse.ArgumentParser(description="GWAS summary statistics QC and standardisation")
    parser.add_argument("--snp-info", default="data/raw/gwas/xiong2025/SnpInfo.tsv", help="Path to SnpInfo.tsv (Xiong 2025)")
    parser.add_argument("--xiong-dir", default="data/raw/gwas/xiong2025/", help="Xiong GWAS directory")
    parser.add_argument("--du-dir", default="data/raw/gwas/du2025/", help="Du GWAS directory")
    parser.add_argument("--bim", default="data/processed/pca/1kg_sas_eur_qc.bim", help="Path to reference BIM file")
    parser.add_argument("--out-dir", default="data/processed/gwas_cleaned/", help="Output directory")
    parser.add_argument("--p-thresh", type=float, default=5e-8, help="P-value filter threshold")
    parser.add_argument("--z-thresh", type=float, default=5.0, help="Z-score threshold for BETA outliers")
    
    args = parser.parse_args()
    
    # argparse converts --foo-bar to args.foo_bar
    out_dir    = args.out_dir
    xiong_dir  = args.xiong_dir
    du_dir     = args.du_dir
    p_thresh   = args.p_thresh
    z_thresh   = args.z_thresh
    snp_info_path = args.snp_info
    bim_path   = args.bim

    os.makedirs(out_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Phase 1: Collect Candidate Coordinates across all traits
    # -------------------------------------------------------------------------
    print("[INFO] Phase 1: Collecting candidate coordinates for significant SNPs...")
    candidate_coords = set()
    
    # Find Xiong files
    xiong_files = []
    for sub in ["Chin", "Lowercheek"]:
        path = os.path.join(xiong_dir, sub, "*.tsv")
        xiong_files.extend(glob.glob(path))
        
    print(f"[INFO] Found {len(xiong_files)} Xiong trait files.")
    
    # Load SnpInfo once for coordinates lookup
    if xiong_files:
        if not os.path.exists(snp_info_path):
            print(f"[ERROR] SnpInfo file not found at '{snp_info_path}'")
            sys.exit(1)
        print(f"[INFO] Reading SnpInfo coordinates from '{snp_info_path}'...")
        snp_info_coords = pd.read_csv(snp_info_path, sep="\t", usecols=["CHR", "BP"])
        
        for trait_file in xiong_files:
            # Read only P column first to find significant rows
            p_df = pd.read_csv(trait_file, sep="\t", usecols=["P"])
            sig_indices = p_df[p_df["P"] < p_thresh].index
            
            # Extract coordinates for significant indices
            for idx in sig_indices:
                row = snp_info_coords.iloc[idx]
                candidate_coords.add((int(row["CHR"]), int(row["BP"])))
                
    # Find Du files
    du_files = glob.glob(os.path.join(du_dir, "*.[tT][sS][vV]")) + \
               glob.glob(os.path.join(du_dir, "*.[tT][xX][tT]")) + \
               glob.glob(os.path.join(du_dir, "*.[cC][sS][vV]"))
    # Filter out status JSON
    du_files = [f for f in du_files if not f.endswith(".json")]
    
    if du_files:
        print(f"[INFO] Found {len(du_files)} Du trait files.")
        # For Du files, we assume standard column names exist inside the files
        for trait_file in du_files:
            try:
                # Quick scan of columns to find P column name
                sample_df = pd.read_csv(trait_file, sep=None, engine='python', nrows=5)
                p_col = next((c for c in sample_df.columns if c.upper() in ['P', 'P-VALUE', 'PVALUE', 'P_VALUE']), None)
                chr_col = next((c for c in sample_df.columns if c.upper() in ['CHR', 'CHROM', 'CHROMOSOME']), None)
                bp_col = next((c for c in sample_df.columns if c.upper() in ['BP', 'POS', 'POSITION', 'BP_GRCH37']), None)
                
                if p_col and chr_col and bp_col:
                    p_df = pd.read_csv(trait_file, sep=None, engine='python', usecols=[chr_col, bp_col, p_col])
                    sig_rows = p_df[p_df[p_col] < p_thresh]
                    for _, row in sig_rows.iterrows():
                        candidate_coords.add((int(row[chr_col]), int(row[bp_col])))
            except Exception as e:
                print(f"[WARN] Error scanning coordinates in Du file '{trait_file}': {e}")
                
    print(f"[INFO] Collected {len(candidate_coords)} unique candidate coordinates.")
    
    if not candidate_coords:
        print("[INFO] No variants passed P-value filter across any trait. Exiting.")
        sys.exit(0)
        
    # -------------------------------------------------------------------------
    # Phase 2: Load Reference BIM Mapping
    # -------------------------------------------------------------------------
    bim_lookup = load_bim_lookup(bim_path, candidate_coords)
    
    # -------------------------------------------------------------------------
    # Phase 3: QC & Clean Xiong Traits
    # -------------------------------------------------------------------------
    if xiong_files:
        print("\n[INFO] Phase 3: Processing Xiong traits...")
        # Load full SnpInfo once
        print(f"[INFO] Loading full SnpInfo from '{snp_info_path}'...")
        snp_info = pd.read_csv(snp_info_path, sep="\t")
        
        for trait_file in xiong_files:
            region = os.path.basename(os.path.dirname(trait_file))
            fname = os.path.basename(trait_file)
            trait_name = f"xiong2025_{region}_{fname.split('.')[0]}"
            print(f"[INFO] Cleaning {trait_name} ('{trait_file}')...")
            
            # Read trait data
            trait_df = pd.read_csv(trait_file, sep="\t")
            
            # Concat SnpInfo and trait statistics (they are row-aligned)
            merged = pd.concat([snp_info, trait_df], axis=1)
            
            # Standardise column names
            # SnpInfo has: CHR, BP, SNP, A1, A2
            # Trait file has: Beta, P
            merged['rsid'] = merged['SNP']
            merged['BETA'] = merged['Beta']
            
            cleaned_df = clean_gwas_df(merged, bim_lookup, p_thresh, z_thresh, trait_name)
            
            if not cleaned_df.empty:
                out_path = os.path.join(out_dir, f"{trait_name}_cleaned.txt")
                cleaned_df.to_csv(out_path, sep="\t", index=False)
                print(f"    Saved cleaned stats to '{out_path}'")
                
    # -------------------------------------------------------------------------
    # Phase 4: QC & Clean Du Traits (if any)
    # -------------------------------------------------------------------------
    if du_files:
        print("\n[INFO] Phase 4: Processing Du traits...")
        for trait_file in du_files:
            fname = os.path.basename(trait_file).split('.')[0]
            trait_name = f"du2025_{fname}"
            print(f"[INFO] Cleaning {trait_name} ('{trait_file}')...")
            
            try:
                raw_df = pd.read_csv(trait_file, sep=None, engine='python')
                
                # Column mapping lookup
                col_map = {}
                for c in raw_df.columns:
                    c_up = c.upper()
                    if c_up in ['CHR', 'CHROM', 'CHROMOSOME']:
                        col_map[c] = 'CHR'
                    elif c_up in ['BP', 'POS', 'POSITION', 'BP_GRCH37']:
                        col_map[c] = 'BP'
                    elif c_up in ['SNP', 'RSID', 'MARKERNAME']:
                        col_map[c] = 'rsid'
                    elif c_up in ['A1', 'ALLELE1', 'EFFECT_ALLELE']:
                        col_map[c] = 'A1'
                    elif c_up in ['A2', 'ALLELE2', 'REFERENCE_ALLELE', 'NONEFFECT_ALLELE']:
                        col_map[c] = 'A2'
                    elif c_up in ['BETA', 'EFFECT', 'LOG_ODDS']:
                        col_map[c] = 'BETA'
                    elif c_up in ['P', 'PVALUE', 'P-VALUE', 'P_VALUE']:
                        col_map[c] = 'P'
                        
                required = {'CHR', 'BP', 'A1', 'A2', 'BETA', 'P'}
                if not required.issubset(set(col_map.values())):
                    missing = required - set(col_map.values())
                    print(f"  [ERROR] Du trait file '{trait_file}' is missing required columns: {missing}. Skipping.")
                    continue
                    
                # Rename columns
                raw_df = raw_df.rename(columns=col_map)
                if 'rsid' not in col_map.values():
                    raw_df['rsid'] = raw_df['CHR'].astype(str) + ":" + raw_df['BP'].astype(str)
                raw_df['SNP'] = raw_df['rsid'] # placeholder, will be resolved by clean_gwas_df
                
                cleaned_df = clean_gwas_df(raw_df, bim_lookup, p_thresh, z_thresh, trait_name)
                
                if not cleaned_df.empty:
                    out_path = os.path.join(out_dir, f"{trait_name}_cleaned.txt")
                    cleaned_df.to_csv(out_path, sep="\t", index=False)
                    print(f"    Saved cleaned stats to '{out_path}'")
                    
            except Exception as e:
                print(f"  [ERROR] Failed to process Du trait file '{trait_file}': {e}")
    else:
        print("\n[INFO] No Du 2025 trait files found to process (status is 'not_yet_downloaded').")
        
    print("\n[INFO] GWAS Cleaning Completed Successfully.")

if __name__ == "__main__":
    main()
