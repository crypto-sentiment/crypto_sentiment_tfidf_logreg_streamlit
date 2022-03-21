from pathlib import Path
import pandas as pd
import pickle
import streamlit as st
import yaml
from inference import model_inference
from utils import get_project_root

# loading config params
project_root: Path = get_project_root()

with open(str(project_root / "config.yml")) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


# loading the model into memory
def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


model = load_model(params["model"]["path_to_model"])


def run_app():

    # headers
    st.title("Cryptonews sentiment")
    st.write("by Yury Kashnitsky")
    st.image(f"static/img/sentiment_icon.jpeg", width=300)

    # get user input from text areas in a Streamlit app
    title = st.text_area("Title", value="BTC drops by 10% today", height=10)

    # process input and run inference
    pred_dict = model_inference(model, title)

    # process predictions
    pred_df = pd.DataFrame.from_dict(pred_dict, orient="index", columns=["pred_score"])
    pred_df["Sentiment"] = pred_df.index
    predicted_class = pred_df["pred_score"].argmax()

    # visualize results
    with open("data/class_map.yml") as f:
        class_map = yaml.load(f, Loader=yaml.FullLoader)

    st.markdown(f"### Predicted class: {class_map[predicted_class]}")
    st.image(f"static/img/{class_map[predicted_class]}_icon.jpeg", width=100)

    st.markdown("Model scores for each class")
    chart_data = pred_df.copy()
    chart_data.index = chart_data.index.map(class_map)
    st.bar_chart(chart_data["pred_score"])


if __name__ == "__main__":
    run_app()
