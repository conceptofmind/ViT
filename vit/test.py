import bentoml

from config import CFG

runner = bentoml.pytorch.load_model("vit:latest")

