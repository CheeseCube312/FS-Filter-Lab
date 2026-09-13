"""
Channel Mixer panel for FS FilterLab.

3×3 slider grid for RGB channel mixing.
"""
from nicegui import ui
from typing import Callable, Optional

from models.constants import CHANNEL_MIXER_RANGE, CHANNEL_MIXER_STEP
from models.core import ChannelMixerSettings


def render_channel_mixer_panel(
    app_state,
    on_change: Optional[Callable] = None,
) -> None:
    """Render the channel mixer 3×3 slider grid inside a card.

    Args:
        app_state: AppState instance.
        on_change: Optional callback fired whenever any slider moves.
    """
    min_val, max_val = CHANNEL_MIXER_RANGE

    def _make_setter(attr: str):
        """Return a callback that writes one mixer attribute to state."""
        def _set(e):
            mixer = app_state.channel_mixer
            setattr(mixer, attr, e.value)
            app_state.channel_mixer = mixer
            if on_change:
                on_change()
        return _set

    def _reset():
        app_state.channel_mixer = ChannelMixerSettings(enabled=app_state.show_channel_mixer)
        # Refresh the UI — replace the panel contents
        panel_container.clear()
        with panel_container:
            _build_sliders()
        if on_change:
            on_change()

    def _build_sliders():
        mixer = app_state.channel_mixer

        # Header row
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Channel Mixer").classes("text-lg font-bold")
            ui.button("Reset", on_click=_reset).props("dense outline size=sm")

        # Column headers
        with ui.grid(columns=4).classes("w-full gap-1"):
            ui.label("Output").classes("font-bold text-sm")
            ui.label("From Red").classes("text-sm italic text-center")
            ui.label("From Green").classes("text-sm italic text-center")
            ui.label("From Blue").classes("text-sm italic text-center")

            # Red output row
            ui.label("Red").classes("font-bold text-sm")
            for attr in ("red_r", "red_g", "red_b"):
                ui.slider(
                    min=min_val, max=max_val, step=CHANNEL_MIXER_STEP,
                    value=getattr(mixer, attr),
                    on_change=_make_setter(attr),
                ).props("dense").classes("w-full")

            # Green output row
            ui.label("Green").classes("font-bold text-sm")
            for attr in ("green_r", "green_g", "green_b"):
                ui.slider(
                    min=min_val, max=max_val, step=CHANNEL_MIXER_STEP,
                    value=getattr(mixer, attr),
                    on_change=_make_setter(attr),
                ).props("dense").classes("w-full")

            # Blue output row
            ui.label("Blue").classes("font-bold text-sm")
            for attr in ("blue_r", "blue_g", "blue_b"):
                ui.slider(
                    min=min_val, max=max_val, step=CHANNEL_MIXER_STEP,
                    value=getattr(mixer, attr),
                    on_change=_make_setter(attr),
                ).props("dense").classes("w-full")

    panel_container = ui.column().classes("w-full")
    with panel_container:
        _build_sliders()
