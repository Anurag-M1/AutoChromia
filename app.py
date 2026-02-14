import io
import os
from collections import OrderedDict

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image


MODEL_INPUT_SIZE = (256, 256)
CHECKPOINT_CANDIDATES = (
    "model.pth",
    "best_model.pth",
    os.path.join("checkpoints", "best_model.pth"),
)


class NotebookColorModel(nn.Module):
    """Model architecture from AutoChromia.ipynb."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def resolve_checkpoint():
    for candidate in CHECKPOINT_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint_obj and all(isinstance(v, torch.Tensor) for v in checkpoint_obj.values()):
            return checkpoint_obj
    raise ValueError(
        "Unsupported checkpoint format. Use a state dict or checkpoint with "
        "'model_state_dict'/'state_dict'."
    )


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return OrderedDict((key[len("module."):], value) for key, value in state_dict.items())
    return state_dict


@st.cache_resource
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint_obj))
    model = NotebookColorModel().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, device


def _normalize_output_channel(channel):
    if torch.min(channel).item() < 0:
        channel = (channel + 1.0) / 2.0
    return channel.clamp(0.0, 1.0)


def colorize_image(image, model, device):
    original_size = image.size
    resized = image.convert("RGB").resize(MODEL_INPUT_SIZE, Image.BILINEAR)
    rgb = np.asarray(resized).astype(np.float32) / 255.0

    l_channel = rgb.mean(axis=2, keepdims=True)
    l_tensor = torch.from_numpy(l_channel).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_rg = model(l_tensor).squeeze(0).cpu()

    pred_r = _normalize_output_channel(pred_rg[0])
    pred_g = _normalize_output_channel(pred_rg[1])
    l_plane = l_tensor.squeeze(0).squeeze(0).cpu()
    pred_b = (3.0 * l_plane - pred_r - pred_g).clamp(0.0, 1.0)

    colorized = torch.stack((pred_r, pred_g, pred_b), dim=2).numpy()
    colorized = (colorized * 255.0).astype(np.uint8)
    return Image.fromarray(colorized).resize(original_size, Image.BILINEAR)


def main():
    st.set_page_config(page_title="AutoChromia", page_icon="🎨", layout="wide")
    st.title("AutoChromia")
    st.caption("Colorize black and white photos using your AutoChromia.ipynb-trained model.")

    checkpoint_path = resolve_checkpoint()
    if checkpoint_path is None:
        st.error("No checkpoint found. Add model.pth or best_model.pth to project root.")
        st.stop()

    st.info(f"Loaded checkpoint: `{checkpoint_path}`")

    try:
        model, device = load_model(checkpoint_path)
    except Exception as exc:
        st.error(f"Failed to load checkpoint: {exc}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a grayscale or old photo",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
    )

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        output_image = colorize_image(input_image, model, device)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(input_image, use_container_width=True)
        with col2:
            st.subheader("Colorized")
            st.image(output_image, use_container_width=True)

        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        st.download_button(
            label="Download Colorized Image",
            data=buffer.getvalue(),
            file_name="colorized.png",
            mime="image/png",
        )

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; font-size:14px;">
            Designed and developed by <b>Anurag Singh</b><br/>
            <a href="https://github.com/anurag-m1" target="_blank">github.com/anurag-m1</a> |
            <a href="https://instagram.com/ca_anuragsingh" target="_blank">instagram.com/ca_anuragsingh</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
