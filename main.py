import streamlit as st
from models import main as evaluate_models
import pandas as pd
import time
import altair as alt

st.set_page_config(layout="wide")

def display_metrics(model_name, accuracy, precision, recall):
    st.write(f"**{model_name} Metrics**")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write("\n---\n")

def main():
     
    st.title("Research Methods Model Analysis")

    st.text("""

    ELECTRA is a new pretraining approach which trains two transformer models: the generator and the discriminator. 
    The generator’s role is to replace tokens in a sequence, and is therefore trained as a masked language model. 
    The discriminator, which is the model we’re interested in, tries to identify which tokens were replaced by the generator in the sequence.
    """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("electra-small-discriminator")
        st.text("""
                Number of layers: 12
                Hidden Size: 256
                Attention heads: 4
                Embedding Size: 128
                Batch Size: 128
                """)
        

    with col2:
        st.subheader("electra-base-discriminator")
        st.text("""
                Number of layers: 12
                Hidden Size: 768
                Attention heads: 12
                Embedding Size: 768
                Batch Size: 256
                """)
        

    with col3:
        st.subheader("electra-large-discriminator")
        st.text("""
                Number of layers: 24
                Hidden Size: 1024
                Attention heads: 16
                Embedding Size: 1024
                Batch Size: 2048
                """)
        

    # Test button to trigger the evaluation
    if st.button("Evaluate Models"):

        progress_text = "Starting evaluation."
        my_bar = st.progress(0, text=progress_text)
        time.sleep(0.50)

        my_bar.progress(10, text='Running models...')
        
        # Call the main method from the models module
        df = evaluate_models()

        my_bar.progress(50, text='Finished running the models.')
        time.sleep(0.50)

        # Display the final DataFrame
        st.write(df)

        # Visualise
        my_bar.progress(70, text='Running the visualisations...')
        time.sleep(0.50)
        visualize_execution_time(df)
        visualize_execution_time_per_sentence(df)
        my_bar.progress(90, text='Finishing visualisations...')
        time.sleep(0.50)
        visualize_mean_memory_consumption(df)
        visualize_memory_consumption_per_sentence(df)
        my_bar.progress(100, text='Analysis complete.')
    

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
