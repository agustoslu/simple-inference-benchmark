# simple-inference-benchmark
Benchmarking different runtimes for vision language models


| Model                   | Runtime     | QPS@1 | TPS@1 | TPQ@1 | VRAMpeak@1 |
| ----------------------- | ----------- | ----- | ----- | ----- | ---------- |
| MiniCPM                 | huggingface | ?     | ?     | ?     | ?          |
| MiniCPM (model.compile) | huggingface | ?     | ?     | ?     | ?          |

# Run the experiment
download the dataset
```bash
python download.py
```

Run the experiment
```bash
python run_minicpm.py
```