import bentoml
from bentoml.io import NumpyNdarray

tag = "vit:latest"

vit_model_runner = bentoml.pytorch.get(tag).to_runner()

svc = bentoml.Service("vit-image-classification", runners=[vit_model_runner])

@svc.api(input = NumpyNdarray(), output = NumpyNdarray())
def classify(input_img):
    result = vit_model_runner.run_batch(input_img)
    return result