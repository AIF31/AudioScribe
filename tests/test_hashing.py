from kado_transcriber.hashing import file_sha256


def test_file_sha256_same_content(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("same", encoding="utf-8")

    assert file_sha256(path) == file_sha256(path)


def test_file_sha256_different_content(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    assert file_sha256(first) != file_sha256(second)
