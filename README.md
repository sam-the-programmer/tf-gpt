# TF-GPT
A TensorFlow implementation of GPT. It implements a stack of decoder blocks for autoregressive text generation, allowing you to make your own foundation models and (smaller) LLMs.

## Usage

To run, simply use the command line:

```powershell
python main.py
```

If you want to train on a custom text file (that fits in RAM) then run the following command, substituting `myDataset.txt` for your text file. If you don't specify your file, it'll train on the [HuggingFace Wikipedia Dataset](https://huggingface.co/datasets/wikitext).

```powershell
python main.py --data="myDataset.txt"
```
