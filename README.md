# OOCC_2021

This repository accompanies the open-access publication *Solar Geoengineering Governance: A Dynamic Framework of Farsighted Coalition Formation (Heyen & Lehtomaa, Oxford Open Climate Change, 2021)*.


Running the code verifies all equilibria and replicates all numerical results discussed in the paper.
The paper can be accessed on the [journal's website](https://academic.oup.com/oocc/article/1/1/kgab010/6370712).
To cite this work, please use:

```
@article{heyen2021solar,
    author = {Heyen, Daniel and Lehtomaa, Jere},
    title = "{Solar Geoengineering Governance: A Dynamic Framework of Farsighted Coalition Formation}",
    journal = {Oxford Open Climate Change},
    year = {2021},
    month = {09},
    issn = {2634-4068},
    doi = {10.1093/oxfclm/kgab010},
    url = {https://doi.org/10.1093/oxfclm/kgab010},
    note = {kgab010},
    eprint = {https://academic.oup.com/oocc/advance-article-pdf/doi/10.1093/oxfclm/kgab010/40392174/kgab010.pdf},
}
```

### Requirements
Running the code relies on minimum dependencies: only `numpy` and `pandas` (and `pytest` for testing) are required.

### Running the code
To replicate all results in the paper, simply run ```python main.py```.
All results will appear in the ```results``` folder.
For testing different player strategies, directly modify the tables in the ```strategy_tables``` folder.
To try out different model parameterizations (discount rates, base temperatures, marginal damages, protocols, etc.), 
modify the ```base_config``` and ```experiment_configs``` variables inside ```main.py```.

# Build for deployment to github pages

cd viz
npm run build

Test locally from root:
python3 -m http.server 8765

Then open http://127.0.0.1:8765.

# Ingest RICE50+ gdx data as payout table

python lib/ingest_payoffs.py --input-dir "/home/frederik/Code/RICE50x/results/Burke" --output "burke_2060" --cutoff-year 2060

# Find equilibrium

Using a payoff table (ingested from RICE run results):
python find_equilibrium.py power_threshold_rice_n3 --payoff-table burke_2060.xlsx

Or use auto-ingest:
python find_equilibrium.py power_threshold_rice_n3 --payoff-table burke_2060.xlsx --auto-ingest

# Orchestrate a full joint run of RICE and coalition model

python multimodel_orchestrator.py --periods 2035-2060 2060-2080 2080-2100 --impact burke --countries usa chn nde --policy bau_impact
