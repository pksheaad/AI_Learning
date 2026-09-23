# Import dependencies
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
from textwrap import indent
import json

import openai
from openai import OpenAI


#----------------------
class PromptEngineering:
    """
    This class will explain the implmentation of zero shot prompt, few shot prompt and chain of thought prompting to understand when and how to use these techniques
        - Zero Shot Prmpt:- for simple task
        - Few More Shot:- Use to improve consistency
        - Chain of Thougth:- Use for reasoning
        - Generate structured output(Json, Table..etc)
        - Compare effectiveness of different techniques
    """

    def __init__(self, api_key:str, model="gpt-3.5-turbo") -> None:
        self.api_key = api_key
        self.model = model

        # create client
        self.client = OpenAI(api_key = api_key)
    
    #Implement zero_shot_prompt
    def zero_shot_prompt(self,prompt:str)->str:
        """
        Zero shot prompt implementation
        """
        messages = [
            {
                "role" : "user",
                "content" : prompt
            }
        ]

        resposne = self.client.chat.completions.create(
            model = self.model,
            messages = messages
        )

        results = resposne.choices[0].message.content

        return results
    
    # Implement few_shot_prompt()
    def few_shot_prompt(self, task:str, examples:List[Dict[str, str]])->str:
        prompt = "Here is the some example\n\n"
        for i, example in enumerate(examples, 1):
            prompt = prompt + f"Example {i}\n"
            prompt = prompt + f"INPUT: {example['input']}\n"
            prompt = prompt + f"OUTPUT: {example['output']}"

        prompt = prompt + f"Now complete the task {task}"

        messages = [
            {
                "role" : "user",
                "content" : prompt
            }
        ]

        response = self.client.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature = 0
        )

        results = response.choices[0].message.content

        return results

    # Implementation of chain of thought
    def chain_of_thought_prompt(self, problem:str)->str:
        prompt = f"{problem}\n\n Please think step by step and explain with an example"

        messages = [
            {
            "role" : "user",
            "content" : prompt
            }   
        ]

        response = self.client.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature = 0.7
        )

        results = response.choices[0].message.content
        return results

    # Implementation of Structured output resposne
    def structured_output_prompt(self, task:str, output_format:str)->str:
        instructions = {
            "JSON" : "Response in a valid JSON formet, maintain the syntax",
            "MARKDOWN" : "Resposne in a MARKDOWN, please main header and properly aliged columns",
            "YAML" : "Resposne in YAML format",
            "CSV" : "Resposne in CSV format"
        }

        prompt = task + f"\n\n {instructions.get(output_format,'JSON')}"

        messages = [
            {
                "role" : "user",
                "content" : prompt
            }
        ]

        response = self.client.chat.completions.create(
            model = self.model,
            messages = messages
        )

        results = response.choices[0].message.content

        return results

    def compare_approaches(self, task:str, examples:List[Dict[str, str]])->Dict[str, str]:
        compare_results = {}
        compare_results['zero-shot'] = self.zero_shot_prompt(prompt = task)

        if examples:
            compare_results['few-shot'] = self.few_shot_prompt(task = task, examples = examples)
        
        compare_results['cot'] = self.chain_of_thought_prompt(problem = task)

        return compare_results




def experiment_1_zero_shot_vs_few_shot():
    api_key = os.getenv(key = "OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"API KEY INVALID OR NOT FOUND")

    prompt = "Classify the sentiment of this review: 'The product is excellent, and shipping was great.'"

    engineer = PromptEngineering(api_key = api_key)
    print("ZERO SHOT LLM RESPOSNE\n")
    print(engineer.zero_shot_prompt(prompt = prompt))

    examples = [
        {
            "input" : "This product goes beyond my expectation",
            "output" : "Positive"
        },
        {
            "input" : "Terrible quality wouldn't recommend",
            "output" : "Negative"
        },
        {
            "input" : "This is OK. Nothing special",
            "output" : "Neutral"
        }
    ]

    print("FEW MORE SHOT PROMPT RESPOSNE \n")
    print(engineer.few_shot_prompt(task = prompt, examples = examples))

def experiment_2_chain_of_thought_prompt():
        api_key = os.getenv(key = "OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"API KEY NOT FOUND OR INVALID")
        
        task = "Calculate the area of sector in a circle"

        engineer = PromptEngineering(api_key = api_key)

        print("CHAIN OF THOUGHT PROCESS\n")
        print(engineer.chain_of_thought_prompt(problem = task)) 

    
def experiment_3_structured_output_prompt():
    api_key = os.getenv(key = "OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"INVALID API KEY")
    
    engineer = PromptEngineering(api_key = api_key)
    
    task = """
    Etract following information from text:
    'Prashant Kumar Singh, age 45' working in Nioda as a Senior Tech Lead'
    Extract: ['name','age','place','designation']
    """
    results = engineer.structured_output_prompt(task = task, output_format = "JSON")
    print(f"JSON Format:\n {results}")

    try:
        parsed = json.loads(results)
        if parsed:
            print(f"PARSED JSON DATA:{json.dumps(parsed, indent = 2)}")
    except json.JSONDecodeError:
        print("NOT A VALID JSON FILE")


def experiment_4_compare_approaches():
    api_key = os.getenv(key = "OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"API KEY IS INVALID")

    problem = """A train travels 120 miles in 2 hours, then stops for 30 minutes,
then travels another 90 miles in 1.5 hours. What is the train's average speed
for the entire journey (including the stop)?"""

    examples = [{
        "input" : "A car travel 100 km in 1 hour, what is its average speed?",
        "output" : "'Speed = Distance/ time', so car average speed is 100/1 = 100 km/hour "

    }]

    engineer = PromptEngineering(api_key = api_key)
    results = engineer.compare_approaches(task = problem, examples = examples)

    print("="*80)
    print("ZERO SHOT RESULT")
    print(f"{results['zero-shot']}")
    print("="*80)
    print("FEW MORE SHOT RESULT")
    print(f"{results['few-shot']}")
    print("="*80)
    print("CAHIN OF TOUGHT RESULTS")
    print(f"{results['cot']}")


# main
if __name__ == "__main__":
    load_dotenv()
    #experiment_1_zero_shot_vs_few_shot()
    #experiment_2_chain_of_thought_prompt()
    #experiment_3_structured_output_prompt()
    experiment_4_compare_approaches()





