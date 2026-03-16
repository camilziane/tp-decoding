# ECoG decoding workshop

This workshop is a hands-on introduction to **decoding finger movements from ECoG signals**.
It walks through a complete pipeline: understanding the dataset, preprocessing the brain signals, building **discrete** and **continuous** decoding models, and evaluating their predictions.
The main support is the notebook [`notebook.ipynb`](/Users/camil/Documents/phd_neuro/tps/tp-decoding/final_notebook.ipynb).

## Quick Start

### Run In Colab

Open the notebook directly in Google Colab:

[Open `notebook.ipynb` in Colab](https://colab.research.google.com/github/camilziane/tp-decoding/blob/main/notebook.ipynb)

### Run Locally With `uv`

Install `uv`:

+ Mac and Linux
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
+ Windows
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

Clone the project:

```bash
git clone https://github.com/camilziane/tp-decoding.git
cd tp-decoding
```

If you do not have `git`, you can also download the project as a ZIP archive from GitHub:

[Download the repository as a ZIP](https://github.com/camilziane/tp-decoding/archive/refs/heads/main.zip)

Then unzip it and open the extracted `tp-decoding-main` folder.

Install the dependencies:

```bash
uv sync
```

Start Jupyter Lab:

```bash
uv run jupyter lab
```

Then open `notebook.ipynb`.
