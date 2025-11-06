from moviepy.editor import VideoFileClip
from napari_animation import Animation
from napari_animation.easing import Easing
from nviz.image import image_set_to_arrays
from nviz.image_meta import generate_ome_xml
from nviz.view import view_ometiff_with_napari


def mp4_to_gif(input_mp4, output_gif, fps=10):
    clip = VideoFileClip(input_mp4)
    clip = clip.set_fps(fps)  # Reduce FPS to control file size
    clip.write_gif(output_gif, loop=0)  # loop=0 makes it loop forever
    return output_gif


def animate_view(
    viewer, output_path_name: str, steps: int = 30, easing: str = "linear", dim: int = 3
):
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
