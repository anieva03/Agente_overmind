import json
from collections import Counter

with open("eval_router.jsonl", encoding="utf-8") as f:
    filas = [json.loads(linea) for linea in f]

etiqueta, n = Counter(fila["expected_output"] for fila in filas).most_common(1)[0]
print(f"Clase mayoritaria: {etiqueta} ({n / len(filas):.1%} de acierto)")