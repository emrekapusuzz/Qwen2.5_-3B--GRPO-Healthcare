# GRPO-Healthcare with Qwen2.5-3B-Instruct

This project implements a healthcare-focused application using the Qwen2.5-3B-Instruct model. By leveraging LoRA-based fine-tuning together with Gradient Reward Policy Optimization (GRPO), the model is trained to generate structured responses that include both detailed, step-by-step reasoning and final diagnostic recommendations. The goal is to assist in medical diagnostics—such as evaluating a 45-year-old patient with chest pain and shortness of breath—by providing comprehensive, traceable outputs.

## Overview

- **Purpose:**  
  Enhance medical diagnostic assistance by training a language model to generate comprehensive reasoning along with final answers for patient cases.

- **Key Components:**  
  - **LoRA Fine-Tuning:** Efficiently adapts the Qwen2.5-3B-Instruct model for the medical domain.
  - **Reward Functions:** Multiple reward functions ensure that outputs follow a strict XML format, accurately convey the reasoning process, and provide correct diagnostic answers.
  - **GRPO Training:** Optimizes the model’s output by rewarding both correctness and high-quality, structured reasoning.
  - **Inference Examples:**  
    - **Without GRPO:** Outputs a detailed diagnostic approach listing history taking, physical examination, diagnostic tests, and risk stratification.
    - **With GRPO:** Provides an XML-formatted response with `<reasoning>` and `<answer>` sections, clarifying the model’s thought process before the final recommendation.

## Features

- **Structured Output:**  
  The GRPO-enhanced model produces responses with explicit XML formatting, separating the chain-of-thought (`<reasoning>`) from the final answer (`<answer>`).

- **Medical Case Reasoning:**  
  Offers detailed diagnostic steps for complex cases, including comprehensive history taking, physical examinations, necessary tests (e.g., ECG, blood tests), and appropriate risk management.

- **Flexible Configuration:**  
  The model can be adapted by adjusting parameters such as the LoRA rank, sequence lengths, and reward functions to meet different resource constraints and output quality requirements.

## Setup and Usage

1. **Installation:**  
   Ensure you have Python 3.8 or higher installed. Set up a virtual environment and install the required libraries:
   - `unsloth`
   - `vllm`
   - `trl`
   - `datasets`
   - `torch`
   - `pillow`

2. **Data Preparation:**  
   The medical dataset is loaded from Hugging Face and transformed into prompt-answer pairs with an XML structure. This format is essential for the reward functions used during GRPO training.

3. **Training:**  
   Configure GRPO training with the defined reward functions (e.g., XML formatting, numeric content checks, and correctness evaluation). Fine-tune the model to generate responses that are both detailed and properly formatted.

4. **Inference:**  
   The project provides two types of inference outputs:
   - **Standard Answer (Without GRPO):**  
     Yields a detailed narrative of the diagnostic approach.
   - **GRPO-Enhanced Answer:**  
     Produces an XML-formatted output that separates the reasoning from the final recommendation:
     ```
     <reasoning>
     [Detailed step-by-step reasoning explaining diagnostic choices...]
     </reasoning>
     <answer>
     [Final recommended diagnostic approach...]
     </answer>
     ```

## Output Comparison

- **Without GRPO:**  
  The model produces a comprehensive diagnostic strategy that includes steps such as history taking, physical examination, diagnostic tests (e.g., ECG, blood tests), and subsequent management.

- **With GRPO:**  
  The response is structured into two distinct parts:
  - `<reasoning>`: Contains the detailed, step-by-step thought process behind the diagnostic approach.
  - `<answer>`: Summarizes the final diagnostic recommendation.

  This structured format enhances clarity and traceability in the model’s decision-making process.
