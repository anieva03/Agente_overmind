import os

import overmind
from datasets import load_dataset
from openai import OpenAI

overmind.init(
    service_name="router-asistent",
    providers=["openai"],
    capability_id=os.environ.get("OVERMIND_CAPABILITY_ID")
)

train=load_dataset("AmazonScience/massive","es-ES" ,split="train")
DOMINIOS=train.features["scenario"].names #los 18 dominios de la base de datos MASSIVE

SISTEMA = (
    "Eres el enrutador de un asistente virtual. Lee la petición del usuario "
    "y responde solo con una de estas etiquetas, sin nada más: "
    + ", ".join(DOMINIOS) + "."
)

cliente = OpenAI()
PROFESOR=os.environ.get("MODELO_PROFESOR", "gpt-4.1-mini")

@overmind.run(intent=lambda peticion: peticion)
def enrutar(peticion: str) -> str:
    respuesta = cliente.chat.completions.create(
        model=PROFESOR,
        temperature=0,
        messages=[
            {"role": "system", "content": SISTEMA},
            {"role": "user", "content": peticion},
        ],
    )
    etiqueta = respuesta.choices[0].message.content.strip().lower()
    overmind.deliver(etiqueta)  # marca el resultado que se puntúa
    return etiqueta

if __name__ == "__main__":
    muestra = train.shuffle(seed=42).select(range(600))
    aciertos = 0
    for fila in muestra:
        prediccion = enrutar(fila["utt"])
        aciertos += prediccion == DOMINIOS[fila["scenario"]]
    print(f"Acierto del profesor: {aciertos / len(muestra):.1%}")