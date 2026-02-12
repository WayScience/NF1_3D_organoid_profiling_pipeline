"""Animation helpers for visualization outputs."""

from typing import Any

from moviepy.editor import VideoFileClip
from napari_animation import Animation
from napari_animation.easing import Easing


def mp4_to_gif(input_mp4: str, output_gif: str, fps: int = 10) -> str:
    """Convert an MP4 file to a looping GIF.

    Parameters
    ----------
    input_mp4 : str
        Path to the source MP4 file.
    output_gif : str
        Path to the output GIF file.
    fps : int, optional
        Frames per second for the GIF, by default 10.

    Returns
    -------
    str
        The output GIF path.
    """
    clip = VideoFileClip(input_mp4)
    clip = clip.set_fps(fps)  # Reduce FPS to control file size
    clip.write_gif(output_gif, loop=0)  # loop=0 makes it loop forever
    return output_gif


def animate_view(
    viewer: Any,
    output_path_name: str,
    steps: int = 30,
    easing: str = "linear",
    dim: int = 3,
) -> str:
    """Animate a napari viewer and save to disk.

    Parameters
    ----------
    viewer : Any
        Napari viewer instance to animate.
    output_path_name : str
        Output file path for the animation.
    steps : int, optional
        Steps per keyframe, by default 30.
    easing : str, optional
        Easing style name, by default "linear".
    dim : int, optional
        Number of displayed dimensions, by default 3.

    Returns
    -------
    str
        The output animation path.
    """
    animation = Animation(viewer)
    if easing == "linear":
        ease_style = Easing.LINEAR
    else:
        raise ValueError(f"Invalid easing style: {easing}")

    viewer.dims.ndisplay = dim
    # rotate around the y-axis
    viewer.camera.angles = (0.0, 0.0, 90.0)  # (z, y, x) axis of rotation
    animation.capture_keyframe(steps=steps, ease=ease_style)

    viewer.camera.angles = (0.0, 180.0, 90.0)
    animation.capture_keyframe(steps=steps, ease=ease_style)

    viewer.camera.angles = (0.0, 360.0, 90.0)
    animation.capture_keyframe(steps=steps, ease=ease_style)

    viewer.camera.angles = (0.0, 0.0, 270.0)
    animation.capture_keyframe(steps=steps, ease=ease_style)

    viewer.camera.angles = (0.0, 0.0, 90.0)
    animation.capture_keyframe(steps=steps, ease=ease_style)

    animation.animate(output_path_name, canvas_only=True)

    return output_path_name
