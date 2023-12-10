import streamlit as st
from models import main as evaluate_models
from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch
import pandas as pd
import time
import psutil
import altair as alt

def display_metrics(model_name, accuracy, precision, recall):
    st.write(f"**{model_name} Metrics**")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write("\n---\n")

def main():
     
    st.title("Research Methods Model Analysis")

    st.text("""

    # Electra Discriminator Performance Evaluation

    Welcome to the Electra Discriminator Performance Evaluation App! This application allows you to assess the performance of Electra discriminators from Google based on key metrics such as execution time, memory consumption, and prediction results.

    ## Overview

    - **Execution Time:** Measure the time taken by each discriminator to process input sentences.
    - **Memory Consumption:** Explore the memory usage of each discriminator during the evaluation.
    - **Prediction Results:** Evaluate the accuracy of predictions made by the Electra discriminators.

    Simply press the "Evaluate Models" button to initiate the assessment, and the app will provide detailed insights into the performance of each model.

    Feel free to customize and analyze the results using the visualization options provided in the app.
    """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("electra-small-discriminator")
        

    with col2:
        st.subheader("electra-base-discriminator")
        

    with col3:
        st.subheader("electra-large-discriminator")
        

    # Test button to trigger the evaluation
    if st.button("Evaluate Models"):
        st.write("Starting the evaluation.")
        
        # Call the main method from the models module
        df = evaluate_models()

        # Display the final DataFrame
        st.write(df)

        # Visualise
        visualize_execution_time(df)
        visualize_execution_time_per_sentence(df)
        visualize_mean_memory_consumption(df)
        visualize_memory_consumption_per_sentence(df)
    

def visualize_execution_time(dataframe):
    st.title("Execution Time Analysis")

    # Check if the DataFrame is empty
    if dataframe.empty:
        st.warning("The provided DataFrame is empty.")
        return

    # Create an Altair chart with a grouped bar chart
    chart = alt.Chart(dataframe).mark_bar().encode(
        x=alt.X('Model:N', title='Model'),
        y=alt.Y('mean(Execution Time):Q', title='Mean Execution Time'),
        color='Model:N',
        tooltip=['Model:N', 'Execution Time:Q']
    ).properties(
        width=600,
        height=400
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=16
    ).configure_title(
        fontSize=20
    )

    # Display the chart
    st.altair_chart(chart, use_container_width=True)

def visualize_execution_time_per_sentence(dataframe):
    st.title("Execution Time per Sentence Analysis")

    # Check if the DataFrame is empty
    if dataframe.empty:
        st.warning("The provided DataFrame is empty.")
        return

    # Create an Altair chart with a line chart
    chart = alt.Chart(dataframe).mark_line().encode(
        x=alt.X('Sentence:N', title='Sentence'),
        y=alt.Y('Execution Time:Q', title='Execution Time'),
        color='Model:N',
        tooltip=['Model:N', 'Sentence:N', 'Execution Time:Q']
    ).properties(
        width=800,
        height=400
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=16
    ).configure_title(
        fontSize=20
    )

    # Display the chart
    st.altair_chart(chart, use_container_width=True)

def visualize_mean_memory_consumption(dataframe):
    st.title("Mean Memory Consumption Analysis")

    # Check if the DataFrame is empty
    if dataframe.empty:
        st.warning("The provided DataFrame is empty.")
        return

    # Create an Altair chart with a grouped bar chart
    chart = alt.Chart(dataframe).mark_bar().encode(
        x=alt.X('Model:N', title='Model'),
        y=alt.Y('mean(Memory Consumption):Q', title='Mean Memory Consumption'),
        color='Model:N',
        tooltip=['Model:N', 'mean(Memory Consumption):Q']
    ).properties(
        width=600,
        height=400
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=16
    ).configure_title(
        fontSize=20
    )

    # Display the chart
    st.altair_chart(chart, use_container_width=True)

def visualize_memory_consumption_per_sentence(dataframe):
    st.title("Memory Consumption per Sentence Analysis")

    # Check if the DataFrame is empty
    if dataframe.empty:
        st.warning("The provided DataFrame is empty.")
        return

    # Create an Altair chart with a line chart
    chart = alt.Chart(dataframe).mark_line().encode(
        x=alt.X('Sentence:N', title='Sentence'),
        y=alt.Y('Memory Consumption:Q', title='Memory Consumption'),
        color='Model:N',
        tooltip=['Model:N', 'Sentence:N', 'Memory Consumption:Q']
    ).properties(
        width=800,
        height=400
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=16
    ).configure_title(
        fontSize=20
    )

    # Display the chart
    st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()
