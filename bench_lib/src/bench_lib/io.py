from abc import ABC, abstractmethod
from pathlib import Path
from llmlib.base_llm import Message
from bench_lib.utils import (
    fill_prompt,
    read_prompt_template,
    mute_video,
    get_transcript,
)
from dataclasses import dataclass
import json


class Input(ABC):
    @abstractmethod
    def prepare(
        self,
        video_path: str | Path,
        meta_data: dict,
        transcript: str | None = None,
        dataset_dir: Path | None = None,
        video_id: str | None = None,
    ) -> dict:
        pass


class OnlyVideo(Input):
    def prepare(
        self,
        video_path: str | Path,
        meta_data: dict,
        transcript: str | None = None,
        dataset_dir: Path | None = None,
        video_id: str | None = None,
    ) -> dict:
        prompt = fill_prompt(meta_data, read_prompt_template())
        return {"messages": [Message(role="user", msg=prompt, video=video_path)]}


class TranscribedVideo(Input):
    def prepare(
        self,
        video_path: str | Path,
        meta_data: dict,
        transcript: str | None = None,
        dataset_dir: Path | None = None,
        video_id: str | None = None,
    ) -> dict:
        assert dataset_dir is not None, "dataset_dir must be provided to mute video"
        transcript = get_transcript(dataset_dir, video_id)
        assert transcript is not None, "Transcript must be provided"
        prompt = fill_prompt(meta_data, read_prompt_template())
        pretty_json = json.dumps(transcript, indent=2, ensure_ascii=False)
        prompt += (
            "\n\nAdditional context:\n"
            "The following is the chunked transcription by the Whisper model:\n"
            "```json\n"
            f"{pretty_json}\n"
            "```\n"
        )
        return {"messages": [Message(role="user", msg=prompt, video=video_path)]}


class MutedVideo(Input):
    def prepare(
        self,
        video_path: str | Path,
        meta_data: dict,
        transcript: str | None = None,
        dataset_dir: Path | None = None,
        video_id: str | None = None,
    ) -> dict:
        assert dataset_dir is not None, "dataset_dir must be provided to mute video"
        transcript = get_transcript(dataset_dir, video_id)
        assert transcript is not None, "Transcript must be provided"
        prompt = fill_prompt(meta_data, read_prompt_template())
        pretty_json = json.dumps(transcript, indent=2, ensure_ascii=False)
        prompt += (
            "\n\nAdditional context:\n"
            "The following is the chunked transcription by the Whisper model:\n"
            "```json\n"
            f"{pretty_json}\n"
            "```\n"
        )
        muted_path = mute_video(dataset_dir, video_path)
        prompt = fill_prompt(meta_data, read_prompt_template())
        return {"messages": [Message(role="user", msg=prompt, video=muted_path)]}


class MutedNoTranscriptVideo(Input):
    def prepare(
        self,
        video_path: str | Path,
        meta_data: dict,
        transcript: str | None = None,
        dataset_dir: Path | None = None,
        video_id: str | None = None,
    ) -> dict:
        assert dataset_dir is not None, "dataset_dir must be provided to mute video"
        muted_path = mute_video(dataset_dir, video_path)
        prompt = fill_prompt(meta_data, read_prompt_template())
        return {"messages": [Message(role="user", msg=prompt, video=muted_path)]}


@dataclass
class Output:
    response: str
    post_id: str | None = None


strategy_map = {
    "only_video": OnlyVideo,
    "transcribed": TranscribedVideo,
    "muted": MutedVideo,
    "muted_no_transcript": MutedNoTranscriptVideo,
}
