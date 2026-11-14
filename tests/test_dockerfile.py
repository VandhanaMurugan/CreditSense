"""P0.4: the Dockerfile at the repository root must package exactly what the service needs to start.
Docker itself is not required to run these checks; they read the file and the repository."""
import fnmatch, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINES = [l.strip() for l in open(os.path.join(ROOT, "Dockerfile")) if l.strip() and not l.strip().startswith("#")]
TEXT = "\n".join(LINES)


def test_base_image_and_start_command():
    assert LINES[0] == "FROM python:3.13-slim"
    assert "pip install -r requirements.txt" in TEXT
    assert LINES[-1].startswith("CMD") and "uvicorn creditsense.api:app" in LINES[-1] and "--host 0.0.0.0" in LINES[-1]
    assert "${PORT:-7860}" in LINES[-1]                       # honours the host's port, defaults to the exposed one
    assert "EXPOSE 7860" in LINES and any(l.startswith("USER ") for l in LINES)      # does not run as root


def test_every_copied_path_exists_and_is_not_ignored():
    ignored = [l.strip() for l in open(os.path.join(ROOT, ".dockerignore")) if l.strip()]
    sources = [l.split()[1] for l in LINES if l.startswith("COPY ")]
    assert sources
    for src in sources:
        assert os.path.exists(os.path.join(ROOT, src)), src
        assert not any(fnmatch.fnmatch(src.rstrip("/"), pat) or src.startswith(pat + "/") for pat in ignored), src


def test_image_contains_what_the_service_loads():
    from creditsense import serve
    copied = [l.split()[1] for l in LINES if l.startswith("COPY ")]
    for needed in (serve._CANDIDATES[-1], serve.GMSC_CSV, os.path.join(ROOT, "creditsense", "api.py"), os.path.join(ROOT, "requirements.txt")):
        rel = os.path.relpath(needed, ROOT)
        assert any(rel == c or rel.startswith(c) for c in copied), rel


def test_requirements_pin_the_model_libraries():
    """The bundle is a pickle: it only loads under the library versions it was built with."""
    req = open(os.path.join(ROOT, "requirements.txt")).read()
    for lib in ("scikit-learn", "xgboost", "shap", "fastapi", "uvicorn", "python-multipart"):
        assert re.search(rf"^{lib}==\d", req, re.M), lib
