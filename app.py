import streamlit as st
import torch
import torchvision
import torchvision.transforms.functional as F
from diffusers import StableDiffusionPipeline
from PIL import ImageDraw, ImageFont
from torch import autocast
from torchvision.utils import make_grid

# List of available models
MODEL_OPTIONS = [
    "maya_model_v1",
    "CompVis/stable-diffusion-v1-4",
]


def add_text_to_image(
    images,
    text,
    text_color="blue",
    outline_color="black",
    font_size=50,
    font_path="arial.ttf",
):
    # Add text to each image
    font = ImageFont.truetype(font_path, size=font_size)

    imgs_with_text = []
    for img in images:
        draw = ImageDraw.Draw(img)
        width, height = img.size

        # Calculate the size of the text
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Calculate the position at which to draw the text to center it
        x = (width - text_width) / 2
        y = (height - text_height) / 2

        # Draw outline
        for adj in range(-2, 3):
            draw.text((x + adj, y), text, font=font, fill=outline_color)
            draw.text((x, y + adj), text, font=font, fill=outline_color)

        # Draw text
        draw.text((x, y), text, fill=text_color, font=font)
        imgs_with_text.append(F.pil_to_tensor(img))

    return imgs_with_text


# Function to generate images
def diff_images(model_name, num_images, prompt, num_inference_steps, text):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    pipe.to("cuda")
    prompt = [prompt] * num_images

    with autocast("cuda"):
        images = pipe(prompt, num_inference_steps=num_inference_steps).images
    imgs = add_text_to_image(
        images,
        text,
        text_color="blue",
        outline_color="black",
        font_size=50,
        font_path="arial.ttf",
    )

    Grid = make_grid(imgs, nrow=1)

    img = torchvision.transforms.ToPILImage()(Grid)
    return img


def main():
    st.title("Diffusion Model Image Generator")

    # Add dropdown widget for model selection
    model_name = st.sidebar.selectbox("Select Diffusion Model", MODEL_OPTIONS)

    # Add widgets to the sidebar
    num_images = st.sidebar.number_input(
        "Number of Images", min_value=1, max_value=10, value=1, step=1
    )

    prompt = st.sidebar.text_area("Text-to-Image Prompt")

    num_inference_steps = st.sidebar.number_input(
        "Number of Inference Steps", min_value=50, max_value=200, value=50, step=1
    )

    text = st.sidebar.text_area("Text to Attach to Images")
    # Generate button
    if st.sidebar.button("Generate Images"):
        if model_name and prompt:
            with st.spinner("Generating images..."):
                generate_images(
                    model_name, num_images, prompt, num_inference_steps, text
                )

        else:
            st.error("Please enter both a model name and a prompt.")


def generate_images(model_name, num_images, prompt, num_inference_steps, text):
    st.subheader("Generated Images:")
    img = diff_images(model_name, num_images, prompt, num_inference_steps, text)
    st.image(img)


if __name__ == "__main__":
    main()
