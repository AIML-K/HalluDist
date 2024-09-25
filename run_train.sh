# Generate data set for training
python src/train/compute_logit.py wow
python src/train/compute_logit.py faithdial
python src/train/compute_logit.py mediqa
python src/train/benchmark/mfact.py wow
python src/train/benchmark/mfact.py faithdial
python src/train/benchmark/mfact.py mediqa

# Weighted training
python src/train/train.py wow logits
python src/train/train.py faithdial logits
python src/train/train.py mediqa logits
python src/train/train.py wow entropy
python src/train/train.py faithdial entropy
python src/train/train.py mediqa entropy

# Benchmark training
python src/train/train.py wow none
python src/train/train.py faithdial none
python src/train/train.py mediqa none
python src/train/benchmark/loss_truncation.py wow
python src/train/benchmark/loss_truncation.py faithdial
python src/train/benchmark/loss_truncation.py mediqa
python src/train/benchmark/mfact_train.py wow
python src/train/benchmark/mfact_train.py faithdial
python src/train/benchmark/mfact_train.py mediqa

# Generate
python src/train/generate.py wow
python src/train/generate.py faithdial
python src/train/generate.py mediqa

# Compute metrics
python src/train/verify.py wow
python src/train/verify.py faithdial
python src/train/verify.py mediqa

python src/train/write_table.py wow
python src/train/write_table.py faithdial
python src/train/write_table.py mediqa