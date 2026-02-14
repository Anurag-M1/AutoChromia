# AutoChromia

AutoChromia is a Streamlit web app that colorizes black-and-white images using a PyTorch checkpoint trained from `AutoChromia.ipynb`.

<img width="1270" height="522" alt="AutoChromia" src="https://github.com/user-attachments/assets/103e0afc-6e6d-4afb-9617-b7e68c4c2830" />

live : https://autochromia.onrender.com
## Features

- Upload grayscale or old photos
- One-click colorization
- Side-by-side input and output preview
- Download the colorized image as PNG

## Project Structure

```text
AutoChromia/
├── app.py
├── best_model.pth
├── AutoChromia.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## Checkpoint Notes

- The app auto-loads checkpoints in this order:
  1. `model.pth`
  2. `best_model.pth`
  3. `checkpoints/best_model.pth`
- Keep at least one of these files in the project root for inference.

## Credits

Designed and developed by Anurag Singh  
GitHub: [github.com/anurag-m1](https://github.com/anurag-m1)  
Instagram: [instagram.com/ca_anuragsingh](https://instagram.com/ca_anuragsingh)
