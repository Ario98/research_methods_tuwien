from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch
import pandas as pd
import time
import psutil

def run_electra(tokenizer, discriminator, sentences):
    discriminator_outputs = []

    for sentence in sentences:
        start_time = time.time()  # Record start time
        process = psutil.Process()  # Get the current process

        inputs = tokenizer.encode(sentence, return_tensors="pt")
        output = discriminator(inputs)

        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time in seconds
        memory_consumption = process.memory_info().rss  # Get memory consumption in bytes

        # Append time and memory information to the output
        output["execution_time"] = execution_time
        output["memory_consumption"] = memory_consumption

        discriminator_outputs.append(output)

    return discriminator_outputs

def evaluate_model(model_uri, sentences):
    discriminator = ElectraForPreTraining.from_pretrained(model_uri)
    tokenizer = ElectraTokenizerFast.from_pretrained(model_uri)

    model_outputs = run_electra(tokenizer, discriminator, sentences)

    data = []
    for sentence_index, output in enumerate(model_outputs):
        result = fact_check_output(output)
        data.append({
            "Sentence": sentences[sentence_index],
            "Model": model_uri,
            "Memory Consumption": output["memory_consumption"],
            "Execution Time": output["execution_time"],
            "Prediction": result,
        })

    df = pd.DataFrame(data)
    return df

def fact_check_output(output):
    predictions = torch.round((torch.sign(output[0]) + 1) / 2)
    if sum(predictions.tolist()[0]) > 0:
        return False
    else:
        return True

def main():
    model_uris = ["google/electra-small-discriminator", "google/electra-base-discriminator", "google/electra-large-discriminator"]
    sentences = ["time is 6 am", "clock shows time as 25pm", "triangle has three corners", "dog barked", "adam said hello [SEP] eve responded how are you"]

    # Collecting data for each model
    data_frames = []

    for model_uri in model_uris:
        df = evaluate_model(model_uri, sentences)
        data_frames.append(df)

    # Combine all DataFrames
    final_df = pd.concat(data_frames, ignore_index=True)

    return final_df

if __name__ == "__main__":
    main()
