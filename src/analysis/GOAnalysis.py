import os
import re
import pandas as pd

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from collections import Counter
from goatools.obo_parser import GODag


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OBO_PATH = os.path.join(ROOT_DIR, "data", "external", "go-basic.obo")

go_dag = GODag(OBO_PATH)

class GOAnalysis:
    def __init__(self, go_sim_mat_dict, go_results_dict, gene_lists, sim_mat_id):
        self.go_sim_mat_dict = go_sim_mat_dict
        self.go_results_dict = go_results_dict
        self.gene_lists = gene_lists
        self.sim_mat_id = sim_mat_id
        self.gene_list = gene_lists[sim_mat_id]
        self.sim_mat = go_sim_mat_dict[sim_mat_id]
        self.go_results = go_results_dict[sim_mat_id]
        self.term_names = (
            self.go_results.loc[self.go_results["Adjusted P-value"] < 0.05, "Term"]
            .dropna()
            .astype(str)
            .str.replace(r"\s*\(.*\)$", "", regex=True)
            .str.strip()
            .tolist()
        )
        self.term_ids = (
            self.go_results.loc[self.go_results["Adjusted P-value"] < 0.05, "Term"]
            .dropna()
            .astype(str)
            .str.extract(r"\((GO:\d+)\)$")[0]
            .tolist()
        )
        self.term_genes = (
            self.go_results.loc[self.go_results["Adjusted P-value"] < 0.05, "Genes"]
            .dropna()
            .astype(str)
            .tolist()
        )
        self.combined_scores = (
            self.go_results.loc[
                self.go_results["Adjusted P-value"] < 0.05, "Combined Score"
            ]
            .dropna()
            .astype(str)
            .tolist()
        )
        self.term_depths = [
            go_dag[term_id].depth if term_id in go_dag else None
            for term_id in self.term_ids
        ]
        self.n_terms = self.sim_mat.shape[0]
        self.find_k_without_singletons().get_reordered_matrix()
        self.get_group_common_genes()
        self.cluster_summary()
        self.group_summary()

    def find_k_without_singletons(self, min_k=10):
    # Handle very small sets immediately
        if self.n_terms < 3:
            self.optimal_k = self.n_terms
            return self

        if self.n_terms < 10:
            self.optimal_k = self.n_terms
            return self

        if not isinstance(self.sim_mat, pd.DataFrame):
            S = pd.DataFrame(self.sim_mat)
        else:
            S = self.sim_mat.copy()

        D = 1.0 - S
        if D.shape[0] < 3:
            raise ValueError("Need at least 3 GO terms.")

        Z = linkage(
            squareform(1.0 - S, checks=False),
            method="ward",
            optimal_ordering=True,
        )

        ks = []
        singleton_rates = {}

        for k in range(2, self.n_terms // 2):
            labs = fcluster(Z, t=k, criterion="maxclust")
            sizes = Counter(labs)
            singleton_rate = sum(1 for v in sizes.values() if v == 1) / len(labs)
            singleton_rates[k] = singleton_rate
            if singleton_rate == 0:
                ks.append(k)

        # Preserve old behavior when ks is non-empty
        if ks:
            self.optimal_k = min_k if max(ks) < min_k else max(ks)
            return self

        # New fallback only for previously crashing cases
        if singleton_rates:
            # smallest singleton rate, then largest k
            self.optimal_k = sorted(singleton_rates, key=lambda k: (singleton_rates[k], -k))[0]
        else:
            # defensive fallback if range(2, self.n_terms // 2) is empty
            self.optimal_k = min(self.n_terms, max(2, min_k))

        return self

    def get_reordered_matrix(self):
        if not isinstance(self.sim_mat, pd.DataFrame):
            S = pd.DataFrame(self.sim_mat)
        else:
            S = self.sim_mat.copy()

        # BYPASS FOR SMALL CLUSTERS
        if self.n_terms < 3:
            # No reordering needed; index remains 0, 1...
            self.reordered_idx = list(range(self.n_terms)) 
            self.S_sorted = S.copy()
            # Assign each term to its own group (1, 2...)
            self.S_sorted["row_color"] = [i + 1 for i in range(self.n_terms)]
            
            print(f"Small cluster detected ({self.n_terms} terms). Skipping hierarchical clustering.")
            return

        # Original logic for N >= 3
        D = 1.0 - S
        Z = linkage(squareform(1.0 - self.sim_mat, checks=False), method="ward", optimal_ordering=True)
        labs = fcluster(Z, t=self.optimal_k, criterion="maxclust")
        self.reordered_idx = leaves_list(Z)

        self.S_sorted = S.iloc[self.reordered_idx, self.reordered_idx].copy()
        self.S_sorted["row_color"] = labs[self.reordered_idx]

        initial_terms = S.shape[0]
        final_terms = self.S_sorted["row_color"].nunique()
        print(
            f"Initial number of GO terms: {initial_terms}, after removing singletons: {final_terms}"
        )

    def get_group_common_genes(self, top_n=5):
        group_counters = {}
        # iterate over reordered indices -> map to S_sorted row positions
        for pos, orig_idx in enumerate(self.reordered_idx):
            group = int(self.S_sorted["row_color"].values[pos])
            genes_str = (
                self.term_genes[orig_idx] if orig_idx < len(self.term_genes) else ""
            )
            if not isinstance(genes_str, str) or genes_str.strip() == "":
                continue
            # normalize separators and split
            parts = re.split(r"[,/;|\t]+", genes_str)
            genes = [p.strip() for p in parts if p.strip()]
            if not genes:
                continue
            counter = group_counters.setdefault(group, Counter())
            for g in genes:
                counter[g] += 1
        # produce top lists
        self.group_common_genes = {
            str(grp): counter.most_common(top_n)
            for grp, counter in group_counters.items()
        }
        return self.group_common_genes

    def _calculate_gene_ratio(self):
        self.go_results["Count"] = self.go_results["Genes"].apply(
            lambda x: len(x.split(";")) if x else 0
        )

        total_input_genes = len(
            self.gene_list
        )  # Use the length of your original cluster list
        self.go_results["Gene Ratio"] = self.go_results["Count"] / total_input_genes

    def cluster_summary(self):
        self._calculate_gene_ratio()
        self.go_sim_cluster_summary = pd.DataFrame(
            {
                "cluster": [self.sim_mat_id] * self.n_terms,
                "group": self.S_sorted["row_color"].values,
                "term_name": self.S_sorted.index.values,
                "term_id": [self.term_ids[i] for i in self.reordered_idx],
                "gene_ratio": [
                    self.go_results.loc[
                        self.go_results["Term"].str.contains(
                            self.term_ids[i], regex=False
                        ),
                        "Gene Ratio",
                    ].values[0]
                    for i in self.reordered_idx
                ],
                "combined_score": [
                    self.go_results.loc[
                        self.go_results["Term"].str.contains(
                            self.term_ids[i], regex=False
                        ),
                        "Combined Score",
                    ].values[0]
                    for i in self.reordered_idx
                ],
                "adjusted_p_value": [
                    self.go_results.loc[
                        self.go_results["Term"].str.contains(
                            self.term_ids[i], regex=False
                        ),
                        "Adjusted P-value",
                    ].values[0]
                    for i in self.reordered_idx
                ],
                "term_depth": [
                    self.term_depths[i] if i < len(self.term_depths) else None
                    for i in self.reordered_idx
                ],
                "gene_count": [
                    self.go_results.loc[
                        self.go_results["Term"].str.contains(
                            self.term_ids[i], regex=False
                        ),
                        "Count",
                    ].values[0]
                    for i in self.reordered_idx
                ],
                "top_genes": [
                    self.go_results.loc[
                        self.go_results["Term"].str.contains(
                            self.term_ids[i], regex=False
                        ),
                        "Genes",
                    ].values[0]
                    for i in self.reordered_idx
                ],
            }
        )
        # fill NaN term_depths with -1 for sorting
        self.go_sim_cluster_summary["term_depth"] = self.go_sim_cluster_summary[
            "term_depth"
        ].fillna(-1)

        # make term_depths integers
        self.go_sim_cluster_summary["term_depth"] = self.go_sim_cluster_summary[
            "term_depth"
        ].astype(int)

        return self

    def group_summary(self):
        # Pre-calculate best terms to ensure correct sorting logic per group
        rep_data = []
        unique_groups = self.S_sorted["row_color"].unique()
        term_count = Counter(self.S_sorted["row_color"])

        for grp in unique_groups:
            # Get all terms belonging to the current group
            grp_terms = self.go_sim_cluster_summary[
                self.go_sim_cluster_summary["group"] == grp
            ]

            stats = []
            for _, row in grp_terms.iterrows():
                # Find the stats in go_results for this term
                match = self.go_results[
                    self.go_results["Term"].str.contains(row["term_id"], regex=False)
                ]
                if not match.empty:
                    m = match.iloc[0]
                    stats.append(
                        {
                            "name": row["term_name"],
                            "ratio": m["Gene Ratio"],
                            "combined_score": m["Combined Score"],
                            "pval": m["Adjusted P-value"],
                            "term_depth": row["term_depth"],
                            "gene_count": m["Count"],
                        }
                    )

            # Sort by: High Gene Ratio, Low P-value, High Count
            best = sorted(
                stats,
                key=lambda x: (
                    -x["ratio"],
                    -x["combined_score"],
                    x["pval"],
                    -x["term_depth"],
                    -x["gene_count"],
                ),
            )[0]
            rep_data.append(best)

        self.go_sim_group_summary = pd.DataFrame(
            {
                "cluster": [self.sim_mat_id] * len(unique_groups),
                "group": list(unique_groups),
                # Count terms in each group explicitly to match unique_groups order
                "term_count": [
                    len(self.S_sorted[self.S_sorted["row_color"] == g])
                    for g in unique_groups
                ],
                "representative_term": [d["name"] for d in rep_data],
                "gene_ratio": [d["ratio"] for d in rep_data],
                "combined_score": [d["combined_score"] for d in rep_data],
                "adjusted_p_value": [d["pval"] for d in rep_data],
                "term_depth": [d["term_depth"] for d in rep_data],
                "gene_count": [d["gene_count"] for d in rep_data],
                "top_genes": list(self.group_common_genes.values()),
                "terms_list": [
                    self.go_sim_cluster_summary.loc[
                        self.go_sim_cluster_summary["group"] == grp, "term_name"
                    ].tolist()
                    for grp in unique_groups
                ],
            }
        )
        return self

class SimplifiedGO(GOAnalysis):
    def __init__(self, go_sim_mat_dict):
        # Only assign the one dictionary you need
        self.go_sim_mat_dict = go_sim_mat_dict
        self.sim_mat_id = list(go_sim_mat_dict.keys())[0]  # Just take the first one
        self.sim_mat = go_sim_mat_dict[self.sim_mat_id]
        self.n_terms = self.sim_mat.shape[0]
        
