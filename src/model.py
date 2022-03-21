from pathlib import Path

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils import get_project_root

# loading config params
project_root: Path = get_project_root()
with open(str(project_root / "config.yml")) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


def initialize_model(params: dict) -> Pipeline:
    """
    Initializes the model, an Sklearn Pipeline with 2 steps: tf-idf and logreg.
    :param params: a dictionary read from the config.yml file
    :return: an Sklearn Pipeline object
    """

    tf_idf_params = params["model"]["tfidf"]
    logreg_params = params["model"]["logreg"]

    # initialize TfIdf, logreg, and the Pipeline with the params from a config file
    text_transformer = TfidfVectorizer(
        stop_words=tf_idf_params["stop_words"],
        ngram_range=eval(tf_idf_params["ngram_range"]),
        lowercase=bool(tf_idf_params["lowercase"]),
        max_features=int(tf_idf_params["max_features"]),
    )

    logreg = LogisticRegression(
        C=int(logreg_params["C"]),
        solver=logreg_params["solver"],
        multi_class=logreg_params["multi_class"],
        random_state=int(logreg_params["random_state"]),
        max_iter=int(logreg_params["max_iter"]),
        n_jobs=int(logreg_params["n_jobs"]),
        fit_intercept=bool(logreg_params["fit_intercept"]),
    )

    model = Pipeline([("tfidf", text_transformer), ("logreg", logreg)])

    return model
