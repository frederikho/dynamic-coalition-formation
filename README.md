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

Standard:
python find_equilibrium.py power_threshold_unequal_power_n3

Save the generated payoff table as an Excel file in `payoff_tables/`:
python find_equilibrium.py power_threshold_unequal_power_n3 --save-payoffs

Using a payoff table (ingested from RICE run results):
python find_equilibrium.py power_threshold_RICE_n3 --payoff-table burke_2060.xlsx

Or use auto-ingest:
python find_equilibrium.py power_threshold_RICE_n3 --payoff-table burke_2060.xlsx --auto-ingest

Using a different sovler:
python find_equilibrium.py power_threshold_RICE_n3 --payoff-table burke_usachnnde_2035-2060.xlsx --solver-approach active_set

Using ordinal ranking:
python find_equilibrium.py power_threshold_RICE_n3 --payoff-table kalkuhl_usachnnde_2035-2060.xlsx --solver-approach ordinal_ranking

Saving all tables:
PYTHONPATH=. python3 scripts/search_ordinal_rankings.py     simple_cycle_usachnnde-65-reduced.xlsx     --scenario power_threshold_RICE_n3     --ranking-order payoff     --weak-orders     --dedup-by strategy --write-all

Allow non-canonical states, used on the further reduced table:
python find_equilibrium.py power_threshold_RICE_n3 --payoff-table simple_cycle_usachn-60-reduced-further.xlsx --allow-non-canonical-states --effectivity-rule free_exit


Finding an equilibrium with weak equalities:
 PYTHONPATH=. python3 scripts/search_ordinal_rankings.py     simple_cycle_usachnnde-60-reduced.xlsx     --scenario power_threshold_RICE_n3  --workers 8         --effectivity-rule free_exit     --ranking-order payoff         --write-all     --dedup-by strategy --weak-orders --weak-equality-solve --max-combinations 50000  --weak-equality-max-vars 30 --progress-every 1000



PYTHONPATH=. python3 scripts/search_ordinal_rankings.py     simple_cycle_usachnnde-100.xlsx     --scenario power_threshold_RICE_n3        --effectivity-rule free_exit     --ranking-order payoff         --write-all     --dedup-by strategy --weak-orders --weak-equality-solve   --workers 8  --weak-equality-max-vars 4  --progress-every 100




LCS:
PYTHONPATH=. python3 scripts/compute_lcs.py simple_cycle_usachn-1.01-reduced-further.xlsx --scenario power_threshold_RICE_n3 --allow-non-canonical-states --effectivity-rule free_exit


# Verify an equilibrium

python lib/verify_cli.py simple_cycle_strategy-100.xlsx

Verify under alternative rules:
python3 lib/verify_cli.py \
    strategy_tables/ordinal_all_simple_cycle_usachn-0-reduced-further/simple_cycle_usachn-0-reduced-further_0001_p010_p111.xlsx \
    --effectivity-rule free_exit


# Repeat find equilibrium

To check stochastic outcome.

python3 scripts/repeat_find_equilibrium.py \
  power_threshold_RICE_n3 \
  --payoff-table simple_cycle_usachnnde-100.xlsx \
  --solver-approach active_set \
  --runs 100 \
  --jobs 12 \
  --timeout-seconds 30

With skipping basins:
python3 scripts/repeat_find_equilibrium.py     power_threshold_RICE_n3     --payoff-table simple_cycle_usachnnde-100.xlsx     --solver-approach active_set     --runs 400     --jobs 12     --timeout 1     --base-seed 1000  --skip-known-basins --stop-on-success

# Orchestrate a full joint run of RICE and coalition model

python multimodel_orchestrator.py --periods 2035-2060 2060-2080 2080-2100 --impact burke --countries usa chn nde --policy bau_impact

With more params:
python3 multimodel_orchestrator.py \
  --max-workers 8 \
  --periods 2035-2060 2060-2080 2080-2100 \
  --impact burke \
  --countries usa chn nde \
  --policy bau_impact \
  --gamma_ineq=0.5 \
  --max_gain=10 \
  --max_damage=0.9 \
  --t_ada_temp=5 \
  --sai_damage_coef=0.1 \
  --fresh

Averaging payoffs, one period:
python3 multimodel_orchestrator.py \
    --max-workers 8 \
    --periods 2035-2100 \
    --impact kalkuhl \
    --countries usa chn nde \
    --policy bau_impact \
    --gamma_ineq=0.5 \
    --max_gain=10 \
    --max_damage=0.9 \
    --t_ada_temp=5 \
    --sai_damage_coef=0.1 \
    --average-payoffs \
    --fresh

## Questions: 

Did I implement the generate_effectivity_heyen_lehtomaa(players, states) correctly? It has the following conditions:
1. Players JOINING a coalition must approve (consent to join)
2. Players LEAVING a coalition must approve (can't be forced out)  
It seemed that this is corresponding to the effectivity in the strategy tables. But did not come out so clear in the paper, particularly the second one. 
... 

After reconsidering, rules are not so bad. But could be formulated in slightly different way. e.g. proposer often does not vote (because already proposed something, so consent is assumed)

derive_effectivity():
Trivially, the proposer must approve the transition,
and is therefore included in the effectivity correspondence.
However, for convenience, we only include the proposer
explicitly in the strategy table when the proposer is the
only approval committee member, and thus can approve
the proposed transition without consulting others.
For every possible proposer, it is always possible to
maintain the status quo without the approval of others.
Therefore, for such a transition, check that the current
proposer is the only member in the effectivity
correspondence. Similarly, any country is allowed to
walk out of its existing coalition.

- It seems we currently assume that only one party deploys G. Makes somewhat sense as we are using a power_threshold of over 0.5 until now and for weak governance we have a free driver effect. But in general, we would also want to allow smaller coalitions than power=0.5 to deploy simultaneously, right? 

- When we are in (FCTWH) and C suggests a move to TWH, who should be on the approval committee? This is in principle like two unilateral exits. So it should be C, F. How does this fit together with the unanimity rule of transition decision making? If we have unanimity (as well as simple majority in this case), C and F have both to agree that both can exit. Is that what we want? 
If three parties want to leave, the three would be on the approval committee. If two want to leave and form a new counter coalition as in (CF)(TWH), now currently TWH would also be on the approval committee. So if TWH disapprove, CF can just leave unilaterally and then forma a coalition afterwards. Is this a good way of modelling? 
An alternative to the current modelling would be that certain types of more complex transitions are outlawed. 

- Can you think of any scenario in which we would get cycling instead of an equilibrium with n=3 players? 

- should power thresholds smaller than 0.5 be allowed?

- is it dynamic or not rather farsighted? farsighted in the sense that players look ahead and we end in the absorbing state. 

## Notes Frederik

- optional: we could define a github action for the npm run build command
- fix the issue occuring for missing files 
