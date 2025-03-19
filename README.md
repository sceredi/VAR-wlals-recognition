# WLASL Recognition and signer classification

This is the project for the course of "Visione Artificiale e Riconoscimento" of the "University of Bologna".
The project aims to classify videos of Word Level American Sign Language into their glosses.
It's also possible to classify each signer using traditional methods and representation learning.

## Project structure
- [data_preprocessing.ipynb](./data_preprocessing.ipynb): shows the preprocessing of the dataset;
- [handcrafted.ipynb](./handcrafted.ipynb): shows the handcrafted feature extraction;
- [wlasl_trad.ipynb](./wlasl_trad.ipynb): is used to show the wlasl classification of the dataset using handcrafted features;
- [wlasl_mediapipe.ipynb](./wlasl_mediapipe.ipynb): is used to show the wlasl classification of the dataset using MediaPipe extracted features;
- [rn-mp-bad-perf-evaluation.ipynb](./rn-mp-bad-perf-evaluation.ipynb): demonstrates that the kaggle solution does not generalize on the problem at hand;
- [mutemotion-wlasl-translation-model-classification.ipynb](./mutemotion-wlasl-translation-model-classification.ipynb): shows a correct classification using the model proposed in the kaggle solution;
- [mutemotion-wlasl-translation-model-regression.ipynb](./mutemotion-wlasl-translation-model-regression.ipynb): shows a correct regression implementation based on the model proposed in the kaggle solution;
- `handcrafted/`: the library regarding all the handcrafted features and the traditional classification;
    - `app/`
        - `dataset/`
        - `features/`
        - `model/`
        - `pca/`
        - `preprocess/`
        - `utilities/`
- `wlasl_mediapipe/`: the library regarding all the MediaPipe features and the classification using MediaPipe;
    - `app/`
        - `dtw/`
        - `mp/`
        - `utils/`
- `wlasl/`
    - `plots/`: containing the confusion matrix plots of the various wlasl classification runs;
    - `results/`: containing the prints of the various wlasl classification runs;

<!-- TODO fix structure after signer classification -->

## How to run

### Requirements
In order to run the project you'll need:
- Python 3.10.13
- ffmpeg (only if you want to execute the data preprocessing step)

### Creation and activation of the python virtual environment
#### Creation

```shell
    python3 -m venv .venv
```

#### Activation
- On linux/macos
    ```bash
    source .venv/bin/activate
    ```
- On windows
    ```shell
    .\.venv\Scrips\activate
    ```
### Dataset download
To download the dataset, if you use bash, you can run the download script:
```shell
    ./download.sh
```
alternatively you can:
- download the zip from [here](https://drive.google.com/file/d/1QbuUJbwrq0D3hU8-sEePb4tJ87t2WA8r/view?usp=drive_link)
- unzip the file into the root directory of the project, so that it has the following structure:
  - `data/`
    - `WLASL_v0.3.json`
    - `missing.txt`
    - `labels.npz`
    - `wlasl_class_list.txt`
    - `videos/`
    - `original_videos_sample/`
    - `hf/`
    - `mp/`

### Download the required libraries
```shell
pip install -r requirements.txt
```



