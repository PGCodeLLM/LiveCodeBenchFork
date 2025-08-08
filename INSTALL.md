# Install 

Follow instructions in README.md

## Manual data setup

- Circumvents https://github.com/LiveCodeBench/LiveCodeBench/issues/108
- Also, code_generation.py was modified to ensure that `release_v{x}` contains only the datapoints in `test{x}.jsonl` (i.e., it is not cumulative anymore).

```
Clone the repo to benchmarks/livecodebench
cd livecodebench
git clone https://huggingface.co/datasets/livecodebench/code_generation_lite data
```

Currently, the latest release is v6. When some newer version is released, it should be just a matter of updating the repo inside `data` (a `git pull` should do it)
