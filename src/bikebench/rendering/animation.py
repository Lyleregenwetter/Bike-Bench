import os
import io
from tqdm import tqdm
import imageio
import cairosvg
from bikebench.rendering.rendering import RenderingEngine  # Adjust if needed


def render_to_animation(
    data,
    mp4_filename,
    renderer=None,
    fps=25,
    max_frames=None,
    rider_dims=None,
):
    """
    Renders each row of a DataFrame using the given renderer and creates an MP4 animation.

    Args:
        data (pd.DataFrame): The dataframe containing rows to render.
        mp4_filename (str): Path where the MP4 will be saved.
        renderer (RenderingEngine or None): If None, a new RenderingEngine will be created.
        fps (int): Frames per second for the MP4.
        max_frames (int or None): If set, limits to the first N frames.
        rider_dims (array-like or None): Optional dimensions for the rider, if applicable.

    Returns:
        BytesIO: In-memory MP4 object (read from saved file).
    """
    if renderer is None:
        renderer = RenderingEngine(number_rendering_servers=1, server_init_timeout_seconds=120)

    output_dir = os.path.dirname(mp4_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    frame_dir = os.path.join(output_dir if output_dir else ".", "frames")
    os.makedirs(frame_dir, exist_ok=True)
    
    png_files = []
    data_iter = data.iterrows()
    if max_frames is not None:
        data_iter = list(data_iter)[:max_frames]

    for i, (_, row) in enumerate(tqdm(data_iter, desc="Rendering frames", total=max_frames or len(data))):
        try:
            res = renderer.render_clip(row, rider_dims=rider_dims)
            svg_bytes = res.image_bytes

            png_path = os.path.join(frame_dir, f"frame_{i:03d}.png")
            cairosvg.svg2png(bytestring=svg_bytes, write_to=png_path)
            png_files.append(png_path)

        except Exception as e:
            print(f"[Warning] Rendering failed for row {i}: {e}")

    with imageio.get_writer(mp4_filename, format="ffmpeg", mode="I", fps=fps, codec="libx264") as writer:
        for png_file in png_files:
            writer.append_data(imageio.imread(png_file))

    # Load and return video as BytesIO
    with open(mp4_filename, "rb") as f:
        mp4_buffer = io.BytesIO(f.read())

    mp4_buffer.seek(0)
    return mp4_buffer
