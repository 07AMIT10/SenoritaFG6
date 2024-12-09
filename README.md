# Fresh Produce Freshness Detection

## Overview

This repository contains a Jupyter Notebook for assessing the freshness of produce using AI-powered image analysis. The system evaluates the visual indicators of fruits and vegetables to determine their freshness on a scale of 1–10 and predicts the expected shelf life in days. It leverages the **FastVisionModel** from the `unsloth` library.

## Features

### 1. Freshness Detection
- Utilizes AI to analyze real-world images of produce, avoiding reliance on stock or generic images.
- Provides a multi-factor assessment, including:
  - **Produce Identification**.
  - **Freshness Score (1–10)**.
  - **Expected Shelf Life (days)**.
  - **Confidence Score**.
  - **Key Visual Indicators** (e.g., discoloration, bruising).

### 2. Database Integration
The notebook outputs data in the following structured format:

| Sl No | Timestamp                  | Produce   | Freshness | Expected Lifespan (Days) |
|-------|----------------------------|-----------|-----------|--------------------------|
| 1     | 2024-11-29T05:14:01+05:30 | Broccoli  | 3         | 5                        |
| 2     | 2024-11-29T05:14:01+05:30 | Onion     | 7         | 12                       |
| 3     | 2024-11-29T05:14:01+05:30 | Papaya    | 1         | 2                        |

### 3. Evaluation Criteria
- **Output Accuracy**: Evaluated against labeled datasets for freshness and shelf life.
- **Technical Innovation**: Combines pre-trained AI models with custom prompts.
- **Practicality**: Real-world utility for grocery stores, suppliers, and consumers.
- **User-Friendliness**: Easy integration into workflows with clearly defined outputs.

## Requirements

### Installation
1. Install required libraries:
    ```bash
    pip install unsloth
    pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
    ```

2. Other dependencies:
    ```bash
    pip install torch pillow pandas ipywidgets
    ```

## Usage

1. **Loading the Model**:
    - The notebook initializes the `FastVisionModel` with pre-trained parameters for produce analysis.
    - Code snippet:
        ```python
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(
            "unsloth/Llama-3.2-11B-Vision-Instruct",
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        ```

2. **Running the Analysis**:
    - Upload an image of the produce and run the notebook to generate:
      - Freshness score.
      - Expected lifespan.
      - Visual cues.

3. **Database Export**:
    - Results are saved in a structured database format for integration into other systems.

## Additional Notes
- This notebook is fully runnable and does not rely on screenshots. Use it directly in [Google Colab](#) or clone the repository for local execution.
- Freshness detection is trained on real-world samples for higher accuracy and reliability.

## License
This repository is licensed under the MIT License. See `LICENSE` for more details.
