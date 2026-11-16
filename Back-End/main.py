from contextlib import asynccontextmanager
import os
import torch
import torch.nn as nn
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel


# Arquitetura do Modelo
class EnemEssayEvaluator(nn.Module):
    def __init__(self, model_name="neuralmind/bert-base-portuguese-cased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 5)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.regressor(self.dropout(cls_output))


# Schemas de Entrada e Saída
class EssayRequest(BaseModel):
    tema: str
    redacao: str


class EssayResponse(BaseModel):
    c1_norma_culta: int
    c2_compreensao_tema: int
    c3_projeto_de_texto: int
    c4_coesao: int
    c5_proposta_intervencao: int
    nota_final: int

# Estado Global
MODEL = None
TOKENIZER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gerenciador de Ciclo de Vida
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, TOKENIZER
    MODEL_PATH = "./saved_model/model_weights.pt"
    TOKENIZER_PATH = "./saved_model/tokenizer"

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Arquivo não encontrado em '{MODEL_PATH}'. "
            "Verifique se a pasta 'saved_model' foi extraída na raiz do projeto."
        )

    print(f"Carregando modelo e tokenizador em {DEVICE}...")
    TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    MODEL = EnemEssayEvaluator().to(DEVICE)
    MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    MODEL.eval()
    print("Modelo carregado e pronto para inferência!")

    yield

    # Limpeza ao desligar o servidor
    MODEL = None
    TOKENIZER = None


app = FastAPI(title="API Avaliadora de Redações ENEM", lifespan=lifespan)

# Libera requisições do Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "online", "device": str(DEVICE)}


@app.post("/api/v1/avaliar", response_model=EssayResponse)
def avaliar_redacao(payload: EssayRequest):
    if len(payload.redacao.split()) < 30:
        raise HTTPException(
            status_code=400,
            detail="A redação é muito curta para ser avaliada (mínimo de 30 palavras)."
        )

    combined_text = f"Tema: {payload.tema} [SEP] Redação: {payload.redacao}"

    inputs = TOKENIZER(
        combined_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        raw_outputs = MODEL(inputs["input_ids"], inputs["attention_mask"]).cpu().numpy()[0]

    # Converte [0, 1] para múltiplos de 40 na escala [0, 200]
    raw_scores = raw_outputs * 200.0
    clipped = np.clip(raw_scores, 0, 200)
    rounded = (np.round(clipped / 40.0) * 40).astype(int)

    return {
        "c1_norma_culta": int(rounded[0]),
        "c2_compreensao_tema": int(rounded[1]),
        "c3_projeto_de_texto": int(rounded[2]),
        "c4_coesao": int(rounded[3]),
        "c5_proposta_intervencao": int(rounded[4]),
        "nota_final": int(np.sum(rounded))
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)