import streamlit as st
from models import main as evaluate_models
from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch
import pandas as pd
import time
import psutil

def display_metrics(model_name, accuracy, precision, recall):
    st.write(f"**{model_name} Metrics**")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write("\n---\n")

def main():
    st.title("Model Evaluation Streamlit App")

    sentences = ["time is 6 am", "clock shows time as 25pm", "triangle has three corners", "dog barked", "adam said hello [SEP] eve responded how are you"]

    # Test button to trigger the evaluation
    if st.button("Evaluate Models"):
        st.write("Evaluating models...")
        
        # Call the main method from the models module
        df = evaluate_models()

        # Display the final DataFrame
        st.write(df)


if __name__ == "__main__":
    main()
