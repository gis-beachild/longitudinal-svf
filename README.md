# longitudinal-svf

Longitudinal training with a subject JSON data config:

```bash
python src/train.py mode=longitudinal_linear data=babofet_bibi train_long=babofet_linear
python src/train.py mode=longitudinal_mlp data=babofet_bibi train_long=babofet_mlp
```

JSON configs use `root_dir`, `train_json`, optional `val_json`, and physical
ages `t0`/`tn`. Each manifest must contain one subject; the training manifest
must include both endpoint ages. Validation can contain held-out sessions and
uses the training endpoints. Label merging follows `merge_labels_0_1` before
conversion to `num_classes` channels. Legacy CSV configs retain endpoint row
indices `t0`/`t1`. Set `train_long.checkpoint=/path/to/last.ckpt` to resume.
