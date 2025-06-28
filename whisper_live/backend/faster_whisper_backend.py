import os
import json
import logging
import threading
import time
import torch
import ctranslate2
from huggingface_hub import snapshot_download

from whisper_live.transcriber.transcriber_faster_whisper import WhisperModel
from whisper_live.backend.base import ServeClientBase


class ServeClientFasterWhisper(ServeClientBase):
    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()


    def as_json_dict(self) -> dict:
        """Return a sanitised mapping of the instance’s attributes.

        Non-serialisable objects (threads, sockets, CTranslate2 models…)
        are string-ified; private/dunder attributes are skipped."""
        def _make_json_safe(val):
            if isinstance(val, (str, int, float, bool)) or val is None:
                return val
            if isinstance(val, (list, tuple)):
                return [_make_json_safe(v) for v in val]
            if isinstance(val, dict):
                return {k: _make_json_safe(v) for k, v in val.items()}
            # Fallback: show the *type* so the client at least knows what’s there
            return f"<{val.__class__.__name__}>"

        return {k: _make_json_safe(v)
                for k, v in self.__dict__.items()
                if not k.startswith(("_", "trans_thread", "websocket", "transcriber"))}


    def __init__(
        self,
            # azinja 2.5
        websocket,
        task="transcribe",
        device=None,
        language=None,
        client_uid=None,
        model="small.en",
        initial_prompt=None,
        vad_parameters=None,
        use_vad=True,
        single_model=False,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=10,
        cache_path="~/.cache/whisper-live/",
        temperature=[0.0],
        log_prob_threshold=-2.0,
        condition_on_previous_text=True,
        word_timestamps=True,
        hallucination_silence_threshold=2.0,
    ):
        """
        Initialize a ServeClient instance.
        The Whisper model is initialized based on the client's language and device availability.
        The transcription thread is started upon initialization. A "SERVER_READY" message is sent
        to the client to indicate that the server is ready.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            task (str, optional): The task type, e.g., "transcribe". Defaults to "transcribe".
            device (str, optional): The device type for Whisper, "cuda" or "cpu". Defaults to None.
            language (str, optional): The language for transcription. Defaults to None.
            client_uid (str, optional): A unique identifier for the client. Defaults to None.
            model (str, optional): The whisper model size. Defaults to 'small.en'
            initial_prompt (str, optional): Prompt for whisper inference. Defaults to None.
            single_model (bool, optional): Whether to instantiate a new model for each client connection. Defaults to False.
            send_last_n_segments (int, optional): Number of most recent segments to send to the client. Defaults to 10.
            no_speech_thresh (float, optional): Segments with no speech probability above this threshold will be discarded. Defaults to 0.45.
            clip_audio (bool, optional): Whether to clip audio with no valid segments. Defaults to False.
            same_output_threshold (int, optional): Number of repeated outputs before considering it as a valid segment. Defaults to 10.

        """
        super().__init__(
            client_uid,
            websocket,
            send_last_n_segments,
            no_speech_thresh,
            clip_audio,
            same_output_threshold,
        )
        self.temperature = temperature,
        self.log_prob_threshold = log_prob_threshold,
        self.condition_on_previous_text = condition_on_previous_text,
        self.word_timestamps = word_timestamps,
        self.hallucination_silence_threshold = hallucination_silence_threshold,

        self.cache_path = cache_path
        self.model_sizes = [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v2", "large-v3", "distil-small.en",
            "distil-medium.en", "distil-large-v2", "distil-large-v3",
            "large-v3-turbo", "turbo"
        ]

        self.model_size_or_path = model
        self.language = "en" if self.model_size_or_path.endswith("en") else language
        self.task = task
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters or {"onset": 0.5}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            major, _ = torch.cuda.get_device_capability(device)
            self.compute_type = "float16" if major >= 7 else "float32"
        else:
            self.compute_type = "int8"

        if self.model_size_or_path is None:
            return
        logging.info(f"Using Device={device} with precision {self.compute_type}")
    
        try:
            if single_model:
                if ServeClientFasterWhisper.SINGLE_MODEL is None:
                    self.create_model(device)
                    ServeClientFasterWhisper.SINGLE_MODEL = self.transcriber
                else:
                    self.transcriber = ServeClientFasterWhisper.SINGLE_MODEL
            else:
                self.create_model(device)
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            self.websocket.send(json.dumps({
                "uid": self.client_uid,
                "status": "ERROR",
                "message": f"Failed to load model: {str(self.model_size_or_path)}"
            }))
            self.websocket.close()
            return

        self.use_vad = use_vad

        # threading
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()
        self.websocket.send(
            json.dumps(
                {
                    "uid": self.client_uid,
                    "message": self.SERVER_READY,
                    "backend": "faster_whisper",
                    "options": self.as_json_dict()
                }
            )
        )


    def create_model(self, device):
        """
        Instantiates a new model, sets it as the transcriber. If model is a huggingface model_id
        then it is automatically converted to ctranslate2(faster_whisper) format.
        """
        model_ref = self.model_size_or_path

        if model_ref in self.model_sizes:
            model_to_load = model_ref
        else:
            logging.info(f"Model not in model_sizes")
            if os.path.isdir(model_ref) and ctranslate2.contains_model(model_ref):
                model_to_load = model_ref
            else:
                local_snapshot = snapshot_download(
                    repo_id = model_ref,
                    repo_type = "model",
                )
                if ctranslate2.contains_model(local_snapshot):
                    model_to_load = local_snapshot
                else:
                    cache_root = os.path.expanduser(os.path.join(self.cache_path, "whisper-ct2-models/"))
                    os.makedirs(cache_root, exist_ok=True)
                    safe_name = model_ref.replace("/", "--")
                    ct2_dir = os.path.join(cache_root, safe_name)

                    if not ctranslate2.contains_model(ct2_dir):
                        logging.info(f"Converting '{model_ref}' to CTranslate2 @ {ct2_dir}")
                        ct2_converter = ctranslate2.converters.TransformersConverter(
                            local_snapshot, 
                            copy_files=["tokenizer.json", "preprocessor_config.json"]
                        )
                        ct2_converter.convert(
                            output_dir=ct2_dir,
                            quantization=self.compute_type,
                            force=False,  # skip if already up-to-date
                        )
                    model_to_load = ct2_dir

        logging.info(f"Loading model: {model_to_load}")
        self.transcriber = WhisperModel(
            model_to_load,
            device=device,
            compute_type=self.compute_type,
            local_files_only=False,
        )

    def set_language(self, info):
        """
        Updates the language attribute based on the detected language information.

        Args:
            info (object): An object containing the detected language and its probability. This object
                        must have at least two attributes: `language`, a string indicating the detected
                        language, and `language_probability`, a float representing the confidence level
                        of the language detection.
        """
        if info.language_probability > 0.5:
            self.language = info.language
            logging.info(f"Detected language {self.language} with probability {info.language_probability}")
            self.websocket.send(json.dumps(
                {"uid": self.client_uid, "language": self.language, "language_prob": info.language_probability}))

    def transcribe_audio(self, input_sample):
        """
        Transcribes the provided audio sample using the configured transcriber instance.

        If the language has not been set, it updates the session's language based on the transcription
        information.

        Args:
            input_sample (np.array): The audio chunk to be transcribed. This should be a NumPy
                                    array representing the audio data.

        Returns:
            The transcription result from the transcriber. The exact format of this result
            depends on the implementation of the `transcriber.transcribe` method but typically
            includes the transcribed text.
        """
        if ServeClientFasterWhisper.SINGLE_MODEL:
            ServeClientFasterWhisper.SINGLE_MODEL_LOCK.acquire()
        result, info = self.transcriber.transcribe(
            # AZINJA 2
            input_sample,
            initial_prompt=self.initial_prompt,
            temperature=self.temperature,
            log_prob_threshold=self.log_prob_threshold,
            condition_on_previous_text=self.condition_on_previous_text,
            word_timestamps=self.word_timestamps,
            hallucination_silence_threshold=self.hallucination_silence_threshold,
            language=self.language,
            task=self.task,
            vad_filter=self.use_vad,
            vad_parameters=self.vad_parameters if self.use_vad else None)
        if ServeClientFasterWhisper.SINGLE_MODEL:
            ServeClientFasterWhisper.SINGLE_MODEL_LOCK.release()

        if self.language is None and info is not None:
            self.set_language(info)
        return result

    def handle_transcription_output(self, result, duration):
        """
        Handle the transcription output, updating the transcript and sending data to the client.

        Args:
            result (str): The result from whisper inference i.e. the list of segments.
            duration (float): Duration of the transcribed audio chunk.
        """
        segments = []
        if len(result):
            self.t_start = None
            last_segment = self.update_segments(result, duration)
            segments = self.prepare_segments(last_segment)

        if len(segments):
            self.send_transcription_to_client(segments)
