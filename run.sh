# Compute Metrics
python src/logit.py gpt2 8
python src/logit.py gpt2-medium 8
python src/logit.py gpt2-large 8
python src/logit.py gpt2-xl 8

python src/logit.py meta-llama/Llama-2-7b-hf 8
python src/logit.py meta-llama/Llama-2-13b-hf 8
python src/logit.py meta-llama/Llama-2-70b-hf 8

##### Batch size for encoder models MUST BE 1!!!
python src/logit.py bert-base-uncased 1
python src/logit.py bert-large-uncased 1
python src/logit.py bert-base-cased 1
python src/logit.py bert-large-cased 1

python src/logit.py albert-base-v2 1
python src/logit.py albert-large-v2 1
python src/logit.py albert-xlarge-v2 1
python src/logit.py albert-xxlarge-v2 1

python src/logit.py roberta-base 1
python src/logit.py roberta-large 1

python src/logit.py facebook/bart-base 8
python src/logit.py facebook/bart-large 8

python src/logit.py t5-small 8
python src/logit.py t5-base 8
python src/logit.py t5-large 8
python src/logit.py t5-3b 8
python src/logit.py t5-11b 8

# Rename files
mv logits/Llama-2-7b-hf.json logits/llama2-7b.json
mv logits/Llama-2-13b-hf.json logits/llama2-13b.json
mv logits/Llama-2-70b-hf.json logits/llama2-70b.json

mv entropy/Llama-2-7b-hf.json entropy/llama2-7b.json
mv entropy/Llama-2-13b-hf.json entropy/llama2-13b.json
mv entropy/Llama-2-70b-hf.json entropy/llama2-70b.json

# Compute Statistics
python src/compute_stat.py logits
python src/compute_stat.py entropy

# Write Table 1
python src/meta_stat.py logits
python src/meta_stat.py entropy

# Plot Figures
python src/fig.py