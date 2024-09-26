Code for our GenerCTC

# Requirements

```
python 3.6.2
torch 1.4.0
tensorflow 1.14.0
```
For detailed dependencies, please refer to requirements.txt

# Get Started

1. From https://huggingface.co/google-bert/bert-base-uncased and https://huggingface.co/FacebookAI/roberta-base and https://huggingface.co/microsoft/MiniLM-L12-H384-uncased, prepare the BERT or RoBERTa or MiniLM pre-trained embeddings, and place the downloaded models in ./model/encoder.
2. Download the full and few-shot datasets from https://github.com/alexa/dialoglue and place them in ./data/datasets.
3. After preprocessing the data, modify the script to train in different modes, for example:
    ```
    python .\scripts\preprocessing\preprocess_banking.py
    python .\scripts\training\train_ctc_banking.py
    ```