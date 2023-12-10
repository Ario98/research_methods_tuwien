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
        execution_time = end_time - start_time  # Calculate execution time
        memory_consumption = process.memory_info().rss  # Get memory consumption

        # Append time and memory information to the output
        output["execution_time"] = execution_time
        output["memory_consumption"] = memory_consumption

        discriminator_outputs.append(output)

    return discriminator_outputs

def fact_check_output(output):
    predictions = torch.round((torch.sign(output[0]) + 1) / 2)
    if sum(predictions.tolist()[0]) > 0:
        return False
    else:
        return True

model_uris = ["google/electra-small-discriminator", "google/electra-base-discriminator", "google/electra-large-discriminator"]

discriminators = []
tokenizers = []

for model_uri in model_uris:
    discriminators.append(ElectraForPreTraining.from_pretrained(model_uri))
    tokenizers.append(ElectraTokenizerFast.from_pretrained(model_uri))

fakes = ["time is 6 am", "clock shows time as 25pm", "triangle has four corners", "cat barked", "adam said hello [SEP] eve responded good night"]
reals = ["time is 6 am", "clock shows time as 25pm", "triangle has three corners", "dog barked", "adam said hello [SEP] eve responded how are you"]

# Collecting data for each model
data = []

for model_index, (discriminator, tokenizer) in enumerate(zip(discriminators, tokenizers)):
    model_uri = model_uris[model_index]
    model_outputs = run_electra(tokenizer, discriminator, reals)

    for sentence_index, output in enumerate(model_outputs):
        result = fact_check_output(output)
        data.append({
            "Sentence": reals[sentence_index],
            "Model": model_uri,
            "Memory Consumption": output["memory_consumption"],
            "Execution Time": output["execution_time"],
            "Prediction": result,
        })

# Create a DataFrame from the collected data
df = pd.DataFrame(data)

# Display the DataFrame
print(df)
