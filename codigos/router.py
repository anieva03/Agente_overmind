import os

import overmind
from datasets import load_dataset
from openai import OpenAI
import time

overmind.init(
    service_name="router-asistent",
    providers=["openai"],
    capability_id=os.environ.get("OVERMIND_CAPABILITY")
)

train = load_dataset("AmazonScience/massive", "es-ES", split="train", trust_remote_code=True)
DOMINIOS=train.features["scenario"].names #los 18 dominios de la base de datos MASSIVE

SISTEMA = (
    "Eres el enrutador de un asistente virtual. Lee la petición del usuario "
    "y responde solo con una de estas etiquetas, sin nada más: "
    + ", ".join(DOMINIOS) + "."
)

cliente = OpenAI(
    base_url="https://api.overmindlab.ai/api/v1",
    api_key=os.environ["OVERMIND_API_KEY"],
)

PROFESOR = os.environ["MODELO_PROFESOR"]

@overmind.run(intent=lambda peticion: peticion)
def enrutar(peticion: str) -> str:
    respuesta = cliente.chat.completions.create(
        model=PROFESOR,
        temperature=0,
        max_tokens=10,
        messages=[
            {"role": "system", "content": SISTEMA},
            {"role": "user", "content": peticion},
        ],
    )

    contenido = respuesta.choices[0].message.content or ""
    if not contenido:
        print(f"Respuesta vacía (finish_reason={respuesta.choices[0].finish_reason}): {peticion!r}")
    etiqueta = contenido.strip().lower()
    overmind.deliver(etiqueta)  # marca el resultado que se puntúa
    return etiqueta

if __name__ == "__main__":
    muestra = train.shuffle(seed=42).select(range(600))
    aciertos = 0
    for fila in muestra:
        prediccion = enrutar(fila["utt"])
        aciertos += prediccion == DOMINIOS[fila["scenario"]]
    print(f"Acierto del profesor: {aciertos / len(muestra):.1%}")