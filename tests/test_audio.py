from audio_transcriber.audio import discover_media_files, is_supported_media


def test_is_supported_media_accepts_known_media_extensions(tmp_path):
    supported = [
        "audio.mp3",
        "video.mp4",
        "recording.wav",
        "meeting.m4a",
        "lecture.flac",
        "podcast.ogg",
    ]

    for filename in supported:
        path = tmp_path / filename
        path.touch()
        assert is_supported_media(path)


def test_is_supported_media_rejects_unknown_extensions_and_directories(tmp_path):
    for filename in ["notes.txt", "image.png", "script.py"]:
        path = tmp_path / filename
        path.touch()
        assert not is_supported_media(path)

    directory = tmp_path / "audio.mp3"
    directory.mkdir()
    assert not is_supported_media(directory)


def test_discover_media_files_returns_sorted_recursive_media_files(tmp_path):
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    root_media = tmp_path / "root.mp3"
    nested_media = nested_dir / "nested.wav"
    ignored = tmp_path / "notes.txt"
    nested_media.touch()
    root_media.touch()
    ignored.touch()

    assert discover_media_files(tmp_path) == [
        nested_media,
        root_media,
    ]


def test_discover_media_files_returns_empty_list_for_missing_or_empty_directory(tmp_path):
    assert discover_media_files(tmp_path) == []
    assert discover_media_files(tmp_path / "missing") == []
