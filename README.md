# ULTR-identifiability

We provide the code for our work: "Identifiability Matters: Revealing the Hidden Recoverable Condition in Unbiased Learning to Rank".

```bash
pip install -r requirements.txt
```

## Dataset

We provide the following datasets for testing:

### Fully simulation datasets

- dataset_fully_simulation/K=1: a connected IG
- dataset_fully_simulation/K=2: an IG with 2 connected components
- dataset_fully_simulation/K=3: an IG with 3 connected components
- dataset_fully_simulation/K=4: an IG with 4 connected components
- dataset_fully_simulation/K=2_node_intervention: a connected IG with applying *node intervention* on K=2 datasets

### Semi-synthetic datasets

We used ULTRA framework (https://github.com/ULTR-Community/ULTRA) to process Yahoo! and Istella-S datasets. You can run the following commands on ULTRA project:

```bash
bash ./example/Yahoo/offline_exp_pipeline.sh
bash ./example/Istella-S/offline_exp_pipeline.sh
```


## Check identifiability with regard to a dataset

Check datasets without context types: 

- Fully simulation dataset

```bash
python identifiability_check.py --data_path "dataset_fully_simulation/K=2"
```

- Yahoo!LETOR

```bash
python identifiability_check.py --data_path "Yahoo_letor/tmp_data"
```

- Check datasets with context types (see the section "Simulate context types").

```bash
python identifiability_check.py --data_path "Yahoo_letor/tmp_data" --context_path "Yahoo_letor/tmp_data/context.pkl"
```

## Test performance on fully simulation datasets

- Test the $K=2$ case:

```bash
python test_fully_simulation.py --data_path "dataset_fully_simulation/K=2" --algorithm "dla"
```

> Algorithm choices: dla / regression_em / two_tower

- Test the $K=2$ case with *node intervention*:

```bash
python test_fully_simulation.py --data_path "dataset_fully_simulation/K=2_node_intervention"
```

- Test the $K=2$ case with *node merging* (here we merge 3 and 4 which is the best strategy, but you can try other strategies):

```bash
python test_fully_simulation.py --data_path "dataset_fully_simulation/K=2" --node_merging_strategies "3-4"
```

- Test the $K=2$ case with *node intervention*, but with random cost:

```bash
python test_fully_simulation.py --data_path "dataset_fully_simulation/K=2" --random_node_intervention
```

- Test another number of clicks:

```bash
python test_fully_simulation.py --data_path "dataset_fully_simulation/K=1" --number_of_clicks 10000
```

## Simulate context types and do *node merging*

- Simulate 5,000 context types on Yahoo! and write to `Yahoo_letor/tmp_data/context.pkl`. It will also do *node merging* and save the merging results.

```bash
python simulate_context_and_node_merging.py --data_path "Yahoo_letor/tmp_data" --context_path "Yahoo_letor/tmp_data/context.pkl" --n_context 5000
```

## Test performance on semi-synthetic datasets

- Test *node merging*:

```bash
python test_semi_synthetic.py --data_path Yahoo_letor/tmp_data --context_path Yahoo_letor/tmp_data/context.pkl
```

- Test *no identification*:

```bash
python test_semi_synthetic.py --data_path Yahoo_letor/tmp_data --context_path Yahoo_letor/tmp_data/context.pkl --no_identification
```

## Citation

Please consider citing the following paper when using our code for your application.

```bibtex
@inproceedings{chen2024identifiability,
  title={Identifiability Matters: Revealing the Hidden Recoverable Condition in Unbiased Learning to Rank},
  author={Mouxiang Chen and Chenghao Liu and Zemin Liu and Zhuo Li and Jianling Sun},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  year={2024}
}
```

