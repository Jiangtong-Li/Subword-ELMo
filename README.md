# Subword ELMo

## Introduction and usage

Github Reporsitory for Arxiv paper titled "Subword ELMo"

Pre-trained model can be downloaded from: 

If you want to use the pretrained model to enhance the representation of your text, you can use the function ```raw2embedding``` in file ```api.py```.

If you want to train your own subword ELMo, first, you should change the configuration in ```config.py``` and then train the model by ```sh run.sh```.(The training process will take too much time, we don't suggest anyone to waste the time and resource.)

## Requirement

    python == 3.6.8
    pytorch == 1.0.0
    typing == 3.7.4.1
    numpy == 1.15.0
    progressbar == 2.5
    overrides == 1.9
    sentencepiece == 0.1.83

We suggest you to use the Anaconda/Miniconda to install all the packages.
