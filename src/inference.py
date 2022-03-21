from pathlib import Path
import yaml
from utils import get_project_root

# loading config params
project_root: Path = get_project_root()

with open(str(project_root / "config.yml")) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


def model_inference(model, input_text):

    pred = model.predict_proba([input_text])

    pred_dict = {}
    for i, pred in enumerate(pred.squeeze()):
        pred_dict[i] = pred

    return pred_dict
