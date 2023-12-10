import streamlit as st
import time

def display_metrics(model_name, accuracy, precision, recall):
    """
    Display metrics in a formatted way.
    """
    st.write(f"**{model_name} Metrics**")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write("\n---\n")

def test_function(progress_bar):
    """
    Function to be called when the test button is clicked.
    """
    for i in range(1, 101):
        time.sleep(0.1)  # Simulating some task that takes time
        progress_bar.progress(i)
    st.write("Test button clicked!")

def main():
    st.title("Model Metrics Dashboard")

    # Progress bar
    progress_bar = st.progress(0)

    # Test button to call the test_function
    if st.button("Test Button"):
        test_function(progress_bar)

    # Placeholder values, replace with actual metrics from your models
    model1_accuracy = 0.85
    model1_precision = 0.78
    model1_recall = 0.92

    model2_accuracy = 0.92
    model2_precision = 0.88
    model2_recall = 0.95

    model3_accuracy = 0.78
    model3_precision = 0.72
    model3_recall = 0.84

    # Display metrics in three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        display_metrics("Model 1", model1_accuracy, model1_precision, model1_recall)

    with col2:
        display_metrics("Model 2", model2_accuracy, model2_precision, model2_recall)

    with col3:
        display_metrics("Model 3", model3_accuracy, model3_precision, model3_recall)

    # Table to display sentences and model outputs
    st.header("Model Outputs for Sentences")
    sentence_data = [
        ("This is a sample sentence.", "Label 1", "Label 2", "Label 3"),
        ("Another example sentence.", "Label 1", "Label 2", "Label 3"),
        # Add more rows with actual data
    ]

    sentences, model1_output, model2_output, model3_output = zip(*sentence_data)

    table_data = {
        "Sentence": sentences,
        "Model 1": model1_output,
        "Model 2": model2_output,
        "Model 3": model3_output,
    }

    st.table(table_data)

if __name__ == "__main__":
    main()
