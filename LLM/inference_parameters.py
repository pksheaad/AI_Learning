""" This class will explain the different Generation Config parameters and its results"""
# importing dependecies
import os
from dotenv import load_dotenv 
from typing import List, Dict

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig
)

class InferenceExplorer:
    """ This class will intatiate the model and tokenizer"""
    def __init__(self, model_name:str = "google/flan-t5-base") -> None:
        self.model_name = model_name

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        

    def generate_with_temperature(self, temperatures: List[float],prompt:str)->List[Dict]:
        """
        This function will take list of different temerature and generate the resposne 
        on the baisi of that with do_sample = True/False
        """
        # create the tokenizer
        results = []
        input_tokens = self.tokenizer(prompt, return_tensors = 'pt')
        for i,temperature in enumerate(temperatures, 1):
            generation_config = GenerationConfig(temperature = temperature)
            with torch.inference_mode():
                response_tokens = self.model.generate(input_tokens['input_ids'],
             generation_config = generation_config)

            model_resposne = self.tokenizer.decode(response_tokens[0], skip_special_tokens = True)

            print(f"{i} : {temperature}")
            print("-"*100)
            print(f"Model Resposne: {model_resposne}")
            results.append({
                "temperature" : temperature,
                "Model_resposne" : model_resposne
            })
        return results

    def generate_greedy(self, prompt:str)->str:

        input = self.tokenizer(prompt, return_tensors = "pt")
        generation_config = GenerationConfig(do_sample = False, num_beams = 1, max_new_tokens = 100)

        with torch.inference_mode():
            output = self.model.generate(
                **input,
                generation_config = generation_config
            )

        response = self.tokenizer.decode(
            output[0],
            skip_special_tokens = True
        )

        return response
        

# Main Function
if __name__ == "__main__":
    load_dotenv()
    # instantiate the class
    explorer = InferenceExplorer()
    temperatures = [0.2, 0.5, 0.7, 1.0]
    prompt = """
Summarize the following text in one sentence:

Artificial intelligence is transforming healthcare,
education and transportation. Machine learning
algorithms can analyze large amounts of data,
identify patterns and help humans make decisions.
"""
    #temperature_results = explorer.generate_with_temperature(temperatures = temperatures, prompt = prompt)
    #print(temperature_results)
    print(explorer.generate_greedy(prompt = prompt))

    


