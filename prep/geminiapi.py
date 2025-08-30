from google import genai
from google.genai import types

import io, os, base64, mimetypes
from PIL import Image as PILImage
from IPython.display import Image, display

def generate_text(prompt: str, api_key: str, 
                    temperature: float = 0.7, 
                    max_output_tokens: int = 200,
                    model: str = "gemini-2.0-flash",
                ):
    client = genai.Client(
        api_key=api_key,
    )
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=20,
            candidate_count=1,
            seed=5,
            max_output_tokens=max_output_tokens,
            stop_sequences=['STOP!'],
            presence_penalty=0.0,
            frequency_penalty=0.0,
        ),
    )
    return response.text

def save_binary_file(filepath: str, data_buffer: bytes):
    """Saves binary data to a file."""
    try:
        with open(filepath, "wb") as f:  # Use "wb" for binary write
            f.write(data_buffer)
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False


def get_base64(data_buffer: bytes) -> str:
    """Converts image data (bytes) to a base64 string."""
    try:
        return base64.b64encode(data_buffer).decode("utf-8")
    except Exception as e:
        print(f"Error converting to base64: {e}")
        return None


def display_base64_image(base64_string: str, image_type: str = "png"):
    """Displays a base64 encoded image in a Jupyter Notebook."""
    try:
        image = Image(data=base64.b64decode(base64_string), format=image_type)
        display(image)
    except Exception as e:
        print(f"Error displaying image: {e}")


def resize_image(image_data: bytes, target_width: int, target_height: int) -> bytes:
    """Resizes an image using Pillow."""
    try:
        img = PILImage.open(io.BytesIO(image_data))  # Open from bytes
        resized_img = img.resize((target_width, target_height))
        # Save the resized image to a BytesIO buffer (in PNG format)
        img_buffer = io.BytesIO()
        resized_img.save(img_buffer, format="PNG") # or another format
        return img_buffer.getvalue()  # Get the bytes data
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None

def generate_image(
        api_key,
        image_prompt,
        model="gemini-2.0-flash-preview-image-generation",
        save_image=False,
        save_path: str = "",
        target_width: int = 1024,
        target_height: int = 768,
        resize_image_flag: bool = False,
    ):
    """
    Generates an image from a prompt, optionally saves it, and optionally resizes it.

    Args:
        api_key: Your Gemini API key.
        image_prompt: The text prompt for image generation.
        model: The Gemini model to use.
        save_image: Whether to save the image to a file.
        save_path: The directory to save the image to.  Must exist.
        target_width: The desired width for resizing (if `resize_image_flag` is True).
        target_height: The desired height for resizing (if `resize_image_flag` is True).
        resize_image_flag: Flag to enable/disable image resizing.  Defaults to False.

    Returns:
        A tuple: (base64_image_string, generated_text_string)
        base64_image_string: Base64 encoded image string (or None if an error occurred).
        generated_text_string: The text generated along with the image (or None).
    """
    gen_img = None  # Initialize before potential use
    gen_txt = None

    client = genai.Client(api_key=api_key)
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=image_prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "IMAGE",
            "TEXT",
        ],
    )

    file_index = 0
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue

        if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            file_name = os.path.join(save_path, f"gemini_gen_image_{file_index}{file_extension}") # Correctly build path

            if resize_image_flag:
                resized_data_buffer = resize_image(data_buffer, target_width, target_height)
                if resized_data_buffer is not None:
                    data_buffer = resized_data_buffer  # Use the resized data

            if save_image:
                if save_binary_file(file_name, data_buffer):
                    print(f"Saved generated image to {file_name}")
                else:
                    print("Error saving image.")
                gen_img = None 
            else:
                gen_img = get_base64(data_buffer) 

            file_index += 1
        else:
            gen_txt = chunk.text
            print(gen_txt)

    return gen_img, gen_txt
