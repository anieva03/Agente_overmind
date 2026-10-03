import json

from datasets import load_dataset

test = load_dataset("AmazonScience/massive", "es-ES", split="test", trust_remote_code=True)

dominios = test.features["scenario"].names

with open("eval_router.jsonl", "w", encoding="utf-8") as f:
    for fila in test.shuffle(seed=7).select(range(300)):
        f.write(json.dumps(
            {"input": fila["utt"], "expected_output": dominios[fila["scenario"]]},
            ensure_ascii=False,
        ) + "\n")