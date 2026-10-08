import pytest

from forgelm.inference import InvalidRequestError
from forgelm.training import JobBusy, JobNotFoundError, TrainingJobStore

TEXT = "to be or not to be, that is the question.\n" * 40

# Small enough that a whole run takes milliseconds, so these are real training runs.
TINY = dict(
    steps=4,
    eval_every=2,
    batch_size=2,
    block_size=8,
    embed_dim=16,
    num_heads=2,
    num_blocks=1,
    device="cpu",
)


@pytest.fixture
def store(tmp_path):
    corpora, checkpoints = tmp_path / "data", tmp_path / "checkpoints"
    corpora.mkdir()
    (corpora / "tiny.txt").write_text(TEXT, encoding="utf-8")
    (corpora / ".hidden.txt").write_text(TEXT, encoding="utf-8")
    (corpora / "notes.md").write_text("not a corpus", encoding="utf-8")
    return TrainingJobStore(corpora, checkpoints)


def test_only_plain_text_files_are_offered_as_corpora(store):
    # A dot-file and a .md file are not training data.
    assert store.list_corpora() == ["tiny.txt"]


def test_an_empty_corpus_folder_is_not_an_error(tmp_path):
    assert TrainingJobStore(tmp_path / "nothing", tmp_path).list_corpora() == []


# --- name validation: the 4a path-traversal lesson, again ------------------------------


@pytest.mark.parametrize(
    "name",
    ["../pyproject.toml", "sub/tiny.txt", "/etc/passwd", ".hidden.txt", "notes.md", ""],
)
def test_a_corpus_must_be_a_plain_file_name(store, name):
    with pytest.raises(InvalidRequestError):
        store.corpus_path(name)


def test_a_corpus_that_does_not_exist_is_an_invalid_request(store):
    with pytest.raises(InvalidRequestError, match="no corpus named"):
        store.corpus_path("missing.txt")


@pytest.mark.parametrize("name", ["../evil.pt", "sub/model.pt", "model.json", ".x.pt", ""])
def test_the_output_must_also_be_a_plain_pt_file_name(store, name):
    with pytest.raises(InvalidRequestError):
        store.checkpoint_path(name)


# --- running a job ---------------------------------------------------------------------


def test_a_finished_job_reports_progress_and_writes_the_checkpoint(store):
    pytest.importorskip("torch")

    job = store.start("tiny.txt", out="gpt.pt", **TINY)
    assert job.status == "running"  # start() returns before training is done
    assert job.steps == 4

    done = store.wait(job.id, timeout=120)

    assert done.status == "done", done.error
    assert [p.step for p in done.progress] == [0, 2, 4]  # eval_every=2, plus the final one
    assert done.val_loss == done.progress[-1].val_loss
    assert done.seconds is not None
    assert (store.checkpoint_dir / "gpt.pt").is_file()


def test_the_checkpoint_can_be_loaded_back(store):
    pytest.importorskip("torch")
    from forgelm.models.minigpt import load_minigpt

    job = store.start("tiny.txt", out="gpt.pt", **TINY)
    store.wait(job.id, timeout=120)

    model, tokenizer = load_minigpt(store.checkpoint_dir / "gpt.pt")

    assert model.block_size == 8
    assert tokenizer.vocab_size == model.vocab_size


def test_a_failing_run_is_recorded_instead_of_killing_the_thread(store):
    pytest.importorskip("torch")

    # A huge learning rate makes the loss diverge, which train_minigpt refuses to continue.
    job = store.start("tiny.txt", out="gpt.pt", **{**TINY, "lr": 1e30})
    done = store.wait(job.id, timeout=120)

    assert done.status == "failed"
    assert "FloatingPointError" in done.error
    assert not (store.checkpoint_dir / "gpt.pt").exists()  # nothing half-written


def test_a_second_run_is_refused_while_one_is_going(store):
    pytest.importorskip("torch")

    first = store.start("tiny.txt", **{**TINY, "steps": 200})
    try:
        with pytest.raises(JobBusy):
            store.start("tiny.txt", **TINY)
    finally:
        store.wait(first.id, timeout=120)

    # Once it has finished, the next run is allowed.
    store.wait(store.start("tiny.txt", **TINY).id, timeout=120)


def test_a_finished_job_is_still_readable_by_id(store):
    pytest.importorskip("torch")

    job = store.start("tiny.txt", **TINY)
    store.wait(job.id, timeout=120)

    assert store.get(job.id).status == "done"
    assert store.latest().id == job.id


def test_an_unknown_job_id_is_not_found(store):
    with pytest.raises(JobNotFoundError):
        store.get("nosuchjob")

    with pytest.raises(JobNotFoundError):
        store.wait("nosuchjob")


def test_latest_is_none_before_anything_started(store):
    assert store.latest() is None


# --- the request is validated before any thread starts ---------------------------------


def test_bad_hyperparameters_are_refused_up_front(store):
    pytest.importorskip("torch")

    with pytest.raises(InvalidRequestError, match="steps must be between"):
        store.start("tiny.txt", **{**TINY, "steps": 10_000})

    with pytest.raises(InvalidRequestError, match="divisible"):
        store.start("tiny.txt", **{**TINY, "embed_dim": 10, "num_heads": 4})

    with pytest.raises(InvalidRequestError, match="corpus"):
        store.start("../pyproject.toml", **TINY)

    assert store.latest() is None  # none of those created a job
