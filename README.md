# DynaConF

This is the code accompying the paper "DynaConF: Dynamic Forecasting of Non-Stationary Time Series".

## Dependencies

Install dependencies:
```bash
conda env create --file environment.yml
```

## Synthetic Data Experiments

Generate synthetic data:
```bash
run/generate_synthetic.sh
```

Run univariate baselines:
```bash
run/synthetic_baselines.sh
```

Run multivariate baselines:
```bash
run/synthetic_baselines_mv.sh
```

Run our models NAR (StatiConF) and NNAR (DynaConF):
```bash
run/synthetic_our.sh
```

Generate the result tables
```bash
run/table_synthetic.sh
```
Results are stored in `./output/synthetic/`.

## Real-World Data (Set 1) Experiments

All the real-world datasets in Set 1 are from GluonTS.

Run our models NAR (StatiConF) and NNAR (DynaConF):
```bash
run/benchmark_our_static.sh
```
and then
```bash
run/benchmark_our_dynamic.sh
```

Generate the result tables
```bash
run/table_benchmark.sh
```
Results are stored in `./output/benchmark/`.

## Real-World Data (Set 2) Experiments

All the real-world datasets in Set 2 are publically available. Information of these datasets are in `./datasets/licenses.csv`. We also include the processed datasets in `./datasets/`, which can be used by copying the unzipped folder to `~/.mxnet/gluon-ts/datasets/`.

Run our models NAR (StatiConF) and NNAR (DynaConF):
```bash
run/benchmark_new_our_static.sh
```
and then
```bash
run/benchmark_new_our_dynamic.sh
```

Run the baseslines
```bash
run/benchmark_new_baselines.sh
```

Generate the result tables
```bash
run/table_benchmark_new.sh
```
Results are stored in `./output/benchmark/`.
