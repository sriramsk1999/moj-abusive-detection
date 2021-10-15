Moj Multilingual Abusive Comment Identification - Challenge
===

Submission for the Moj Abusive Comment Detection Challenge, hosted on [Kaggle](https://www.kaggle.com/c/iiitd-abuse-detection-challenge/).

Setup
---

1. Download the dataset and extract it the root directory.
2. Create a virtual environment:
``` sh
python -m venv env # virtual env
pip install -r requirements.txt
source env/bin/activate
```
3. Change `index` in `main.py` to choose which model of the ensemble to train/test.
4. Run `python main.py`
5. `utils.py` contains helper functions for caching and ensembling.

**Note: BERT models are large, GPU with 16GB VRAM required. Batch size can be reduced if training on 8GB GPU.**

