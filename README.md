## Input

After cloning, extract `data.zip` in the repository root to create the `data/` directory:

```bash
python -m zipfile -e data.zip .
```

`data/pep_gpcr_train_val.csv` is used for training and validation. `data/pep_gpcr_test.csv` is used only for testing. Each CSV must contain `SMILES` (molecule), `Protein` (protein sequence), and `Y` (binary label: 0 or 1). Other columns are ignored by the model.

`weights/pepti.pt` provides the starting model weights for DPS fine-tuning.

## Output

Training writes `runs/train/best.pt`, `validation.csv` (metrics by epoch), and `run.json` (run settings). Testing writes `runs/test/test_results.csv` with `index` (row in the test CSV), `prob` (positive-class probability), and `label` (true label).

## Datasets

`mol_gpcr.csv` contains small-molecule–GPCR binding data.

`pep_target_nodup.csv` contains peptide–target binding data, including targets outside the GPCR family. Peptide–target pairs that overlap with or are highly similar to PepGI have been removed.

`pep_gpcr_train_val.csv` and `pep_gpcr_test.csv` contain peptide–GPCR binding data, split into training/validation and test sets.

## Environment

```text
Python          3.11.16
PyTorch         2.4.0+cu121
DGL             2.4.0+cu121
DGLLife         0.3.2
RDKit           2024.9.6
NumPy           2.2.6
pandas          3.0.6
scikit-learn    1.9.1
```



## Training

```bash
python src/run.py train \
    --stage pepgi \
    --data data/pep_gpcr_train_val.csv \
    --test-already-split \
    --init weights/pepti.pt \
    --output "runs/train" \
    --device cuda:0 --batch-size 256 --workers 4 \
    --epochs 100 --lr 5e-5 --keep-ratio 0.6 \
    --patience 10 --stop-metric auroc
```

The output directory must not already exist.

## Evaluate

```bash
python src/run.py evaluate \
    --stage pepgi \
    --data data/pep_gpcr_test.csv \
    --all-data \
    --init "runs/train/best.pt" \
    --output "runs/test" \
    --device cuda:0 --batch-size 256 --workers 4
```

