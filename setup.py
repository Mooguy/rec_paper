from setuptools import setup, find_packages

setup(
    name="bioinfo_analysis_project",
    version="0.1.0",
    # Tell pip that the root packages are inside the 'src' folder
    package_dir={"": "."}, 
    packages=find_packages(where="."),
    install_requires=[
            "pandas",
            "numpy",
            "scipy",
            "matplotlib",
            "seaborn",
            "scanpy",
            "anndata",
            "xgboost",
            "optuna",
            "scikit-learn",
            "statsmodels",
            "gseapy",
            "pyarrow",       
            "pingouin",
            "tqdm",
            "umap-learn",
            "goatools",
        ],
)