# WLASL Recognition and signer classification

This is the project for the course of "Visione Artificiale e Riconoscimento" of the "University of Bologna".
The project aims to classify videos of Word Level American Sign Language into their glosses.
It's also possible to classify each signer using traditional methods and representation learning.

## Project structure
- [1_data_preprocessing.ipynb](./1_data_preprocessing.ipynb): shows the preprocessing of the dataset;
- [2_handcrafted.ipynb](./2_handcrafted.ipynb): shows the handcrafted feature extraction;
- [3_wlasl_trad.ipynb](./3_wlasl_trad.ipynb): is used to show the wlasl classification of the dataset using handcrafted features;
- [4_wlasl_mediapipe.ipynb](./4_wlasl_mediapipe.ipynb): is used to show the wlasl classification of the dataset using MediaPipe extracted features;
- [5_nn_mp_bad_perf_evaluation.ipynb](./5_nn_mp_bad_perf_evaluation.ipynb): demonstrates that the kaggle solution does not generalize on the problem at hand;
- [6_mutemotion_wlasl_translation_model_classification.ipynb](./6_mutemotion_wlasl_translation_model_classification.ipynb): shows a correct classification using the model proposed in the kaggle solution;
- [7_mutemotion_wlasl_translation_model_regression.ipynb](./7_mutemotion_wlasl_translation_model_regression.ipynb): shows a correct regression implementation based on the model proposed in the kaggle solution;
- [8_signer_trad.ipynb](./8_signer_trad.ipynb): face classification using traditional classifiers and handcrafted features;
- [9_signer_cnn.ipynb](./9_signer_cnn.ipynb): face classification using a simple CNN model;
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
    - `frames_no_bg/`
    - `original_videos_sample/`
    - `hf/`
    - `mp/`

### Download the required libraries
```shell
pip install -r requirements.txt
```



