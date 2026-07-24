"""CustomTkinter GUI and threaded application runner."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, replace
from typing import Callable

import customtkinter as ctk
import cv2

from src.action_manager import ActionManager
from src.camera import CameraCapture, CameraError
from src.config import ConfigManager, GestureSettings, MouseSettings, Settings
from src.gesture_detector import GestureDetector, GestureType
from src.hand_tracker import HandLandmarkIndex, HandTracker
from src.mouse_controller import MouseController

PREVIEW_WINDOW_NAME = "Hand Gesture Mouse - Camera Preview"


@dataclass
class ApplicationStatus:
    """Thread-safe snapshot of runtime application status."""

    running: bool = False
    mouse_enabled: bool = False
    fps: float = 0.0
    hand_count: int = 0
    gesture: str = "none"
    error_message: str | None = None


class GestureApplication:
    """
    Runs the camera, tracking, and mouse control loop in a background thread.

    Exposes start/stop controls and live settings updates for the GUI.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        logger: logging.Logger,
    ) -> None:
        """
        Initialize the gesture application runner.

        Args:
            config_manager: Configuration manager for the project.
            logger: Application logger instance.
        """
        self._config_manager = config_manager
        self._logger = logger
        self._settings = config_manager.get_settings()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = ApplicationStatus()

        self._camera: CameraCapture | None = None
        self._hand_tracker: HandTracker | None = None
        self._mouse_controller: MouseController | None = None
        self._gesture_detector: GestureDetector | None = None
        self._action_manager: ActionManager | None = None

    @property
    def is_running(self) -> bool:
        """Return True if the background processing thread is active."""
        with self._lock:
            return self._status.running

    def get_status(self) -> ApplicationStatus:
        """
        Return a copy of the current application status.

        Returns:
            ApplicationStatus snapshot.
        """
        with self._lock:
            return ApplicationStatus(
                running=self._status.running,
                mouse_enabled=self._status.mouse_enabled,
                fps=self._status.fps,
                hand_count=self._status.hand_count,
                gesture=self._status.gesture,
                error_message=self._status.error_message,
            )

    def get_settings(self) -> Settings:
        """
        Return the current in-memory settings.

        Returns:
            Active Settings instance.
        """
        with self._lock:
            return self._settings

    def apply_settings(self, settings: Settings) -> None:
        """
        Apply updated settings to running components.

        Args:
            settings: New settings values from the GUI.
        """
        with self._lock:
            self._settings = settings
            if self._mouse_controller is not None:
                self._mouse_controller.update_settings(settings.mouse)
            if self._gesture_detector is not None:
                self._gesture_detector.update_settings(settings.gestures)

    def start(self) -> None:
        """Start the background camera and gesture processing thread."""
        with self._lock:
            if self._status.running:
                return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="GestureApplicationThread",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and release all resources."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

        with self._lock:
            self._status.running = False
            self._status.mouse_enabled = False
            self._status.fps = 0.0
            self._status.hand_count = 0
            self._status.gesture = GestureType.NONE.value

    def toggle_mouse(self) -> bool:
        """
        Toggle mouse control on or off.

        Returns:
            New mouse enabled state.
        """
        with self._lock:
            if self._mouse_controller is None:
                return False

            enabled = self._mouse_controller.toggle()
            if not enabled and self._gesture_detector and self._action_manager:
                self._gesture_detector.reset()
                self._action_manager.reset()
            self._status.mouse_enabled = enabled
            return enabled

    def set_mouse_enabled(self, enabled: bool) -> None:
        """
        Explicitly enable or disable mouse control.

        Args:
            enabled: Desired mouse control state.
        """
        with self._lock:
            if self._mouse_controller is None:
                return

            if enabled:
                self._mouse_controller.enable()
            else:
                self._mouse_controller.disable()
                if self._gesture_detector and self._action_manager:
                    self._gesture_detector.reset()
                    self._action_manager.reset()
            self._status.mouse_enabled = enabled

    def _run_loop(self) -> None:
        """Background loop for capture, tracking, preview, and mouse control."""
        settings = self.get_settings()
        self._camera = CameraCapture(settings.camera, logger=self._logger)
        self._hand_tracker = HandTracker(
            settings.hand_tracking,
            project_root=self._config_manager.project_root,
            logger=self._logger,
        )
        self._mouse_controller = MouseController(settings.mouse, logger=self._logger)
        self._gesture_detector = GestureDetector(settings.gestures)
        self._action_manager = ActionManager(logger=self._logger)

        try:
            self._camera.start()
        except CameraError as error:
            self._logger.error("Camera startup failed: %s", error)
            with self._lock:
                self._status.error_message = str(error)
                self._status.running = False
            self._cleanup_components()
            return

        with self._lock:
            self._status.running = True
            self._status.error_message = None
            self._status.mouse_enabled = False

        self._logger.info("Gesture application thread started.")

        try:
            while not self._stop_event.is_set():
                frame = self._camera.read_frame()
                if frame is None:
                    continue

                tracking_result = self._hand_tracker.process(frame)
                self._hand_tracker.draw(frame, tracking_result)

                primary_hand = tracking_result.primary_hand
                self._draw_control_point(frame, primary_hand)

                gesture_result = self._gesture_detector.detect(primary_hand)

                with self._lock:
                    mouse_enabled = self._mouse_controller.is_enabled
                    self._status.fps = self._camera.fps
                    self._status.hand_count = tracking_result.hand_count
                    self._status.gesture = gesture_result.gesture.value

                if mouse_enabled:
                    self._mouse_controller.update_from_hand(primary_hand)
                    self._action_manager.process(gesture_result, mouse_enabled=True)
                else:
                    if primary_hand is None:
                        self._gesture_detector.reset()
                    self._action_manager.process(gesture_result, mouse_enabled=False)

                self._draw_status_overlays(
                    frame,
                    fps=self._camera.fps,
                    hand_count=tracking_result.hand_count,
                    mouse_enabled=mouse_enabled,
                    gesture=gesture_result.gesture,
                )

                cv2.imshow(PREVIEW_WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    self._logger.info("Preview closed via keyboard.")
                    break
                if key in (ord("m"), ord("M")):
                    self.toggle_mouse()
        except cv2.error as error:
            self._logger.error("OpenCV error in application loop: %s", error)
            with self._lock:
                self._status.error_message = str(error)
        finally:
            self._cleanup_components()
            with self._lock:
                self._status.running = False
                self._status.mouse_enabled = False
            cv2.destroyAllWindows()
            self._logger.info("Gesture application thread stopped.")

    def _cleanup_components(self) -> None:
        """Release camera, tracker, and mouse resources."""
        if self._action_manager is not None:
            self._action_manager.reset()
        if self._mouse_controller is not None:
            self._mouse_controller.disable()
        if self._hand_tracker is not None:
            self._hand_tracker.close()
        if self._camera is not None:
            self._camera.stop()

        self._camera = None
        self._hand_tracker = None
        self._mouse_controller = None
        self._gesture_detector = None
        self._action_manager = None

    @staticmethod
    def _draw_control_point(frame, hand) -> None:
        """
        Highlight the index finger tip on the preview frame.

        Args:
            frame: BGR image to annotate in place.
            hand: Detected hand data, or None.
        """
        if hand is None:
            return

        tip_x, tip_y = hand.pixel_landmarks[HandLandmarkIndex.INDEX_FINGER_TIP]
        cv2.circle(frame, (tip_x, tip_y), 12, (0, 255, 255), 2)
        cv2.circle(frame, (tip_x, tip_y), 4, (0, 255, 255), -1)

    @staticmethod
    def _draw_status_overlays(
        frame,
        fps: float,
        hand_count: int,
        mouse_enabled: bool,
        gesture: GestureType,
    ) -> None:
        """
        Draw runtime status text on the preview frame.

        Args:
            frame: BGR image to annotate in place.
            fps: Current frames per second.
            hand_count: Number of detected hands.
            mouse_enabled: Whether mouse control is active.
            gesture: Current detected gesture.
        """
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        hand_label = f"Hands: {hand_count}" if hand_count else "No hand detected"
        hand_color = (0, 255, 0) if hand_count else (0, 0, 255)
        cv2.putText(
            frame,
            hand_label,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            hand_color,
            2,
            cv2.LINE_AA,
        )

        mouse_label = "Mouse: ON" if mouse_enabled else "Mouse: OFF"
        mouse_color = (0, 255, 255) if mouse_enabled else (180, 180, 180)
        cv2.putText(
            frame,
            mouse_label,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            mouse_color,
            2,
            cv2.LINE_AA,
        )

        if mouse_enabled and gesture != GestureType.NONE:
            gesture_label = f"Gesture: {gesture.value.replace('_', ' ').title()}"
        else:
            gesture_label = "Gesture: —"
        cv2.putText(
            frame,
            gesture_label,
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )


class HandGestureAppGUI(ctk.CTk):
    """
    Main settings and control window for the hand gesture mouse application.

    Provides start/stop controls, live setting sliders, theme selection,
    and a real-time status dashboard.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        logger: logging.Logger,
    ) -> None:
        """
        Initialize the main GUI window.

        Args:
            config_manager: Configuration manager for the project.
            logger: Application logger instance.
        """
        settings = config_manager.get_settings()
        super().__init__()

        self._config_manager = config_manager
        self._logger = logger
        self._settings = settings
        self._app = GestureApplication(config_manager, logger)
        self._slider_labels: dict[str, ctk.CTkLabel] = {}

        self._apply_theme(settings.gui.theme)
        self._configure_window(settings)
        self._build_layout()
        self._start_status_polling()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _apply_theme(self, theme: str) -> None:
        """
        Apply CustomTkinter appearance mode.

        Args:
            theme: Theme name from settings ("dark" or "light").
        """
        appearance = "dark" if theme.lower() == "dark" else "light"
        ctk.set_appearance_mode(appearance)
        ctk.set_default_color_theme("blue")

    def _configure_window(self, settings: Settings) -> None:
        """
        Configure main window title and dimensions.

        Args:
            settings: Application settings.
        """
        self.title(settings.app.name)
        self.geometry(f"{settings.gui.window_width}x{settings.gui.window_height}")
        self.minsize(820, 620)

    def _build_layout(self) -> None:
        """Build all GUI sections and controls."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkLabel(
            self,
            text="AI Hand Gesture Mouse Control",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        header.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        content = ctk.CTkScrollableFrame(self)
        content.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)

        self._build_control_section(content)
        self._build_status_section(content)
        self._build_mouse_settings_section(content)
        self._build_gesture_settings_section(content)
        self._build_appearance_section(content)

    def _build_control_section(self, parent: ctk.CTkScrollableFrame) -> None:
        """Build start/stop and mouse toggle buttons."""
        section = ctk.CTkFrame(parent)
        section.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        section.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkLabel(
            section,
            text="Controls",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, padx=12, pady=(12, 8), sticky="w")

        ctk.CTkButton(
            section,
            text="Start",
            command=self._on_start_clicked,
        ).grid(row=1, column=0, padx=12, pady=(0, 12), sticky="ew")

        ctk.CTkButton(
            section,
            text="Stop",
            command=self._on_stop_clicked,
        ).grid(row=1, column=1, padx=12, pady=(0, 12), sticky="ew")

        self._mouse_toggle_button = ctk.CTkButton(
            section,
            text="Enable Mouse",
            command=self._on_toggle_mouse_clicked,
        )
        self._mouse_toggle_button.grid(row=1, column=2, padx=12, pady=(0, 12), sticky="ew")

    def _build_status_section(self, parent: ctk.CTkScrollableFrame) -> None:
        """Build live status labels."""
        section = ctk.CTkFrame(parent)
        section.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        section.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(
            section,
            text="Status",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 8), sticky="w")

        self._status_running_label = ctk.CTkLabel(section, text="Running: No")
        self._status_running_label.grid(row=1, column=0, padx=12, pady=4, sticky="w")

        self._status_fps_label = ctk.CTkLabel(section, text="FPS: 0.0")
        self._status_fps_label.grid(row=1, column=1, padx=12, pady=4, sticky="w")

        self._status_hands_label = ctk.CTkLabel(section, text="Hands: 0")
        self._status_hands_label.grid(row=2, column=0, padx=12, pady=4, sticky="w")

        self._status_gesture_label = ctk.CTkLabel(section, text="Gesture: none")
        self._status_gesture_label.grid(row=2, column=1, padx=12, pady=4, sticky="w")

        self._status_mouse_label = ctk.CTkLabel(section, text="Mouse: OFF")
        self._status_mouse_label.grid(row=3, column=0, padx=12, pady=(4, 12), sticky="w")

        self._status_error_label = ctk.CTkLabel(section, text="", text_color="#ff6666")
        self._status_error_label.grid(row=3, column=1, padx=12, pady=(4, 12), sticky="w")

    def _build_mouse_settings_section(self, parent: ctk.CTkScrollableFrame) -> None:
        """Build mouse movement setting sliders."""
        section = ctk.CTkFrame(parent)
        section.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        section.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            section,
            text="Mouse Settings",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, padx=12, pady=(12, 8), sticky="w")

        self._add_slider(
            section,
            row=1,
            label="Smoothing Factor",
            from_=0.05,
            to=1.0,
            initial=self._settings.mouse.smoothing_factor,
            callback=self._on_smoothing_changed,
        )
        self._add_slider(
            section,
            row=2,
            label="Sensitivity",
            from_=0.5,
            to=2.0,
            initial=self._settings.mouse.sensitivity,
            callback=self._on_sensitivity_changed,
        )
        self._add_slider(
            section,
            row=3,
            label="Dead Zone",
            from_=0.0,
            to=0.1,
            initial=self._settings.mouse.dead_zone,
            callback=self._on_dead_zone_changed,
        )

    def _build_gesture_settings_section(self, parent: ctk.CTkScrollableFrame) -> None:
        """Build gesture detection setting sliders."""
        section = ctk.CTkFrame(parent)
        section.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        section.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            section,
            text="Gesture Settings",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, padx=12, pady=(12, 8), sticky="w")

        self._add_slider(
            section,
            row=1,
            label="Pinch Threshold",
            from_=0.02,
            to=0.12,
            initial=self._settings.gestures.pinch_threshold,
            callback=self._on_pinch_threshold_changed,
        )
        self._add_slider(
            section,
            row=2,
            label="Drag Hold (ms)",
            from_=100,
            to=800,
            initial=float(self._settings.gestures.drag_hold_ms),
            callback=self._on_drag_hold_changed,
            is_integer=True,
        )
        self._add_slider(
            section,
            row=3,
            label="Scroll Sensitivity",
            from_=200,
            to=1600,
            initial=self._settings.gestures.scroll_sensitivity,
            callback=self._on_scroll_sensitivity_changed,
        )

    def _build_appearance_section(self, parent: ctk.CTkScrollableFrame) -> None:
        """Build theme selector and save button."""
        section = ctk.CTkFrame(parent)
        section.grid(row=4, column=0, sticky="ew")
        section.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            section,
            text="Appearance & Settings",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, padx=12, pady=(12, 8), sticky="w")

        theme_row = ctk.CTkFrame(section, fg_color="transparent")
        theme_row.grid(row=1, column=0, padx=12, pady=4, sticky="ew")
        theme_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(theme_row, text="Theme").grid(row=0, column=0, padx=(0, 12), sticky="w")
        self._theme_option = ctk.CTkOptionMenu(
            theme_row,
            values=["dark", "light"],
            command=self._on_theme_changed,
        )
        self._theme_option.set(self._settings.gui.theme)
        self._theme_option.grid(row=0, column=1, sticky="ew")

        ctk.CTkButton(
            section,
            text="Save Settings",
            command=self._on_save_settings_clicked,
        ).grid(row=2, column=0, padx=12, pady=(12, 12), sticky="ew")

    def _add_slider(
        self,
        parent: ctk.CTkFrame,
        row: int,
        label: str,
        from_: float,
        to: float,
        initial: float,
        callback: Callable[[float], None],
        is_integer: bool = False,
    ) -> None:
        """
        Add a labeled slider control to a settings section.

        Args:
            parent: Parent frame widget.
            row: Grid row index inside the parent.
            label: Setting label text.
            from_: Minimum slider value.
            to: Maximum slider value.
            initial: Initial slider value.
            callback: Function called when the slider value changes.
            is_integer: Whether to display the value as an integer.
        """
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.grid(row=row, column=0, padx=12, pady=6, sticky="ew")
        container.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(container, text=label).grid(row=0, column=0, padx=(0, 12), sticky="w")

        value_label = ctk.CTkLabel(container, text=self._format_slider_value(initial, is_integer))
        value_label.grid(row=0, column=2, padx=(12, 0), sticky="e")
        self._slider_labels[label] = value_label

        def on_slide(value: float) -> None:
            parsed = int(round(value)) if is_integer else round(value, 3)
            value_label.configure(text=self._format_slider_value(parsed, is_integer))
            callback(float(parsed))

        slider = ctk.CTkSlider(container, from_=from_, to=to, command=on_slide)
        slider.set(initial)
        slider.grid(row=0, column=1, sticky="ew")

    @staticmethod
    def _format_slider_value(value: float, is_integer: bool) -> str:
        """
        Format a slider value for display.

        Args:
            value: Numeric slider value.
            is_integer: Whether to format as an integer.

        Returns:
            Formatted value string.
        """
        if is_integer:
            return str(int(value))
        return f"{value:.3f}"

    def _push_settings(self) -> None:
        """Apply current in-memory settings to the running application."""
        self._app.apply_settings(self._settings)

    def _on_smoothing_changed(self, value: float) -> None:
        """Handle smoothing factor slider changes."""
        self._settings = replace(
            self._settings,
            mouse=replace(self._settings.mouse, smoothing_factor=value),
        )
        self._push_settings()

    def _on_sensitivity_changed(self, value: float) -> None:
        """Handle sensitivity slider changes."""
        self._settings = replace(
            self._settings,
            mouse=replace(self._settings.mouse, sensitivity=value),
        )
        self._push_settings()

    def _on_dead_zone_changed(self, value: float) -> None:
        """Handle dead zone slider changes."""
        self._settings = replace(
            self._settings,
            mouse=replace(self._settings.mouse, dead_zone=value),
        )
        self._push_settings()

    def _on_pinch_threshold_changed(self, value: float) -> None:
        """Handle pinch threshold slider changes."""
        self._settings = replace(
            self._settings,
            gestures=replace(self._settings.gestures, pinch_threshold=value),
        )
        self._push_settings()

    def _on_drag_hold_changed(self, value: float) -> None:
        """Handle drag hold duration slider changes."""
        self._settings = replace(
            self._settings,
            gestures=replace(self._settings.gestures, drag_hold_ms=int(value)),
        )
        self._push_settings()

    def _on_scroll_sensitivity_changed(self, value: float) -> None:
        """Handle scroll sensitivity slider changes."""
        self._settings = replace(
            self._settings,
            gestures=replace(self._settings.gestures, scroll_sensitivity=value),
        )
        self._push_settings()

    def _on_theme_changed(self, theme: str) -> None:
        """Handle theme selection changes."""
        self._apply_theme(theme)
        self._settings = replace(
            self._settings,
            gui=replace(self._settings.gui, theme=theme),
        )

    def _on_start_clicked(self) -> None:
        """Start the background gesture application."""
        self._app.start()
        self._logger.info("Application started from GUI.")

    def _on_stop_clicked(self) -> None:
        """Stop the background gesture application."""
        self._app.stop()
        self._logger.info("Application stopped from GUI.")

    def _on_toggle_mouse_clicked(self) -> None:
        """Toggle mouse control from the GUI."""
        if not self._app.is_running:
            self._status_error_label.configure(text="Start the app first.")
            return

        enabled = self._app.toggle_mouse()
        self._update_mouse_button(enabled)
        self._logger.info("Mouse control toggled from GUI: %s", enabled)

    def _on_save_settings_clicked(self) -> None:
        """Persist current settings to the local override file."""
        try:
            self._config_manager.save_local_settings(self._settings)
            self._status_error_label.configure(text="Settings saved.", text_color="#66ff99")
            self._logger.info("Settings saved to local override file.")
        except OSError as error:
            self._status_error_label.configure(text=f"Save failed: {error}", text_color="#ff6666")
            self._logger.error("Failed to save settings: %s", error)

    def _update_mouse_button(self, enabled: bool) -> None:
        """
        Update mouse toggle button label.

        Args:
            enabled: Current mouse control state.
        """
        label = "Disable Mouse" if enabled else "Enable Mouse"
        self._mouse_toggle_button.configure(text=label)

    def _start_status_polling(self) -> None:
        """Begin periodic refresh of status labels."""
        self._refresh_status_labels()

    def _refresh_status_labels(self) -> None:
        """Refresh dashboard labels from application status."""
        status = self._app.get_status()

        self._status_running_label.configure(
            text=f"Running: {'Yes' if status.running else 'No'}"
        )
        self._status_fps_label.configure(text=f"FPS: {status.fps:.1f}")
        self._status_hands_label.configure(text=f"Hands: {status.hand_count}")
        self._status_gesture_label.configure(text=f"Gesture: {status.gesture}")
        self._status_mouse_label.configure(text=f"Mouse: {'ON' if status.mouse_enabled else 'OFF'}")
        self._update_mouse_button(status.mouse_enabled)

        if status.error_message:
            self._status_error_label.configure(
                text=status.error_message,
                text_color="#ff6666",
            )

        self.after(200, self._refresh_status_labels)

    def _on_close(self) -> None:
        """Handle window close by stopping background threads."""
        self._app.stop()
        self.destroy()

    def run(self) -> None:
        """Start the GUI main loop."""
        self.mainloop()


def launch_gui(config_manager: ConfigManager, logger: logging.Logger) -> None:
    """
    Create and run the main application GUI.

    Args:
        config_manager: Configuration manager for the project.
        logger: Application logger instance.
    """
    app = HandGestureAppGUI(config_manager, logger)
    app.run()
