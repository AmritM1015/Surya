# Easy Inference

Use this folder for the simplest Surya flow:
1. choose date window,
2. download only required hourly files,
3. run rollout inference (default debug rollout = 2),
4. save one `prediction.nc`.

## Quick start

```bash
source .venv/bin/activate
bash easy_inference/run_easy_inference.sh
```

Non-interactive defaults:

```bash
bash easy_inference/run_easy_inference.sh --no-prompt
```

## Config

Edit `easy_inference/config_easy.yaml`.

- Normal users: edit only the top `user:` section.
- Advanced users: optional changes in `advanced:`.

## Metrics Notebook

Use `easy_inference/compare_prediction_groundtruth.ipynb` to compare `prediction.nc` vs GT and compute:
- overall metrics (`MSE`, `RMSE`, `MAE`, `bias`, `max_abs_error`)
- per-channel metrics
- per-step metrics
- visual prediction vs ground-truth plots
