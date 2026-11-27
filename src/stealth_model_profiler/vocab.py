"""Candidate vocabularies and local token counting.

Two ways to turn a probe into an expected token count:

``hf``        a published ``tokenizer.json`` from Hugging Face (open vocabularies).
``tiktoken``  OpenAI's published BPE ranks.

Neither is authoritative for a served model — vendors fork vocabularies, and a
gateway may count with something else entirely — so these predictions are scored
probabilistically against *measured* counts, never treated as ground truth.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

CACHE_DIR = Path(os.environ.get("MPROF_CACHE", Path.home() / ".cache" / "model-profiler"))


@dataclass(frozen=True)
class VocabSpec:
    key: str
    lab: str
    family: str
    kind: str          # "hf" | "tiktoken"
    source: str        # HF repo id, or tiktoken encoding name
    note: str = ""
    filename: str = "tokenizer.json"
    revision: str = ""
    sha256: str = ""


# Several entries per lab on purpose: a lab's vocabulary drifts between
# generations, and "closest generation" is itself useful evidence.
VOCABS: List[VocabSpec] = [
    # Z.ai / Zhipu
    VocabSpec("glm-5.3", "zai", "GLM-5.3", "hf", "zai-org/GLM-5.3",
              revision="ba32f8c19c8738e2c57c71798c601d477a7bf844",
              sha256="19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"),
    VocabSpec("glm-5.2", "zai", "GLM-5.2", "hf", "zai-org/GLM-5.2",
              revision="cf457fa734ab149ffef225f80893eb38c6ff5cdc",
              sha256="19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"),
    VocabSpec("glm-5", "zai", "GLM-5", "hf", "zai-org/GLM-5",
              revision="c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2",
              sha256="19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"),
    VocabSpec("glm-4.6", "zai", "GLM-4.6", "hf", "zai-org/GLM-4.6",
              revision="be72194883d968d7923a07e2f61681ea9a2826d1",
              sha256="9340665016419c825c4bdabbcc9acc43b7ca2c68ce142724afa829abb1be5efd"),
    VocabSpec("glm-4.5", "zai", "GLM-4.5", "hf", "zai-org/GLM-4.5",
              revision="cbb2c7cfb52fa128a9660cb1a7a78e017899e115",
              sha256="9340665016419c825c4bdabbcc9acc43b7ca2c68ce142724afa829abb1be5efd"),
    # Alibaba / Qwen
    VocabSpec("qwen3", "alibaba", "Qwen3", "hf", "Qwen/Qwen3-235B-A22B",
              revision="8efa61729e24bd65b1d152b5ab5409052aa80e65",
              sha256="aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    VocabSpec("qwen3-coder", "alibaba", "Qwen3-Coder", "hf", "Qwen/Qwen3-Coder-480B-A35B-Instruct",
              revision="9d90cf8fca1bf7b7acca42d3fc9ae694a2194069",
              sha256="19564a48c4f71a2a1b937cce34c737a1e662b171c5f5d7edf641a15cd896f07d"),
    # DeepSeek
    VocabSpec("deepseek-v3.2", "deepseek", "DeepSeek-V3.2", "hf", "deepseek-ai/DeepSeek-V3.2",
              revision="a7e62ac04ecb2c0a54d736dc46601c5606cf10a6",
              sha256="cd050be35cae877f8f0aa847f45aa87e23835a56ca32b29b28545597852784e5"),
    VocabSpec("deepseek-v3", "deepseek", "DeepSeek-V3", "hf", "deepseek-ai/DeepSeek-V3",
              revision="e815299b0bcbac849fa540c768ef21845365c9eb",
              sha256="621ac2e32d0dba658404412318818aaa8ce8cda492e59830109d8da6b517fb41"),
    # Mistral
    VocabSpec("mistral-large", "mistral", "Mistral Large", "hf",
              "mistralai/Mistral-Large-Instruct-2411",
              revision="ba78820945ae22361b0274cf0ae6d696c967c1a4",
              sha256="2482d54bc351bc5f9ccc899ffd51b8d6cbe0fc1987e8a81a2d7c3670c0b666fc"),
    VocabSpec("mistral-nemo", "mistral", "Mistral NeMo", "hf", "mistralai/Mistral-Nemo-Instruct-2407",
              revision="04d8a90549d23fc6bd7f642064003592df51e9b3",
              sha256="e11c71726323d33da7b8d6f6f269f1988931c0a52b7122bcdd8c05042974e0db"),
    # Xiaomi / MiMo
    VocabSpec("mimo", "xiaomi", "MiMo", "hf", "XiaomiMiMo/MiMo-7B-RL",
              revision="7c7707596b0ef1032ef82970d7a4a484ca457c4d",
              sha256="c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
    # OpenAI
    VocabSpec("o200k", "openai", "GPT-4o/5 family", "tiktoken", "o200k_base"),
    VocabSpec("cl100k", "openai", "GPT-4/3.5 family", "tiktoken", "cl100k_base"),
    VocabSpec("o200k-harmony", "openai", "gpt-oss / Harmony", "hf", "openai/gpt-oss-120b",
              revision="b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
              sha256="0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3"),
    # Meta (gated upstream; several public mirrors publish the same ranks)
    VocabSpec("llama3", "meta", "Llama 3.x", "hf", "NousResearch/Meta-Llama-3-8B",
              revision="315b20096dc791d381d514deb5f8bd9c8d6d3061",
              sha256="e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7"),
    VocabSpec("llama4", "meta", "Llama 4", "hf", "unsloth/Llama-4-Scout-17B-16E-Instruct",
              revision="afd8e498c87bda51c7ea8ec68ea2f7c066e6340b",
              sha256="172c9eb4beafc72601690da3ccfcede5c2e6806a8d5ec1fca33e22acea8023a4"),
    # Google
    VocabSpec("gemma3", "google", "Gemma 3", "hf", "unsloth/gemma-3-27b-it",
              revision="7a5a3053dbd5d1d58e48159e87b9df2fc545a49a",
              sha256="4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795"),
    VocabSpec("gemma2", "google", "Gemma 2", "hf", "unsloth/gemma-2-27b-it",
              revision="c3d7bab17f14fbda4ad59f1dec764eb7562c7e55",
              sha256="3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922"),
]

_BY_KEY: Dict[str, VocabSpec] = {v.key: v for v in VOCABS}


class Vocab:
    """A loaded vocabulary that can count strings."""

    def __init__(self, spec: VocabSpec, counter: Callable[[str], int]):
        self.spec = spec
        self._counter = counter

    def count(self, text: str) -> int:
        return self._counter(text)

    def delta(self, base: str, probe: str) -> int:
        """Tokens added by ``probe`` when appended to ``base``.

        The boundary matters: BPE merges across the join, so this must be computed
        on the concatenation, never as ``count(probe)``.
        """
        return self._counter(base + probe) - self._counter(base)

    @property
    def key(self) -> str:
        return self.spec.key


def _download(url: str, dest: Path, digest: str) -> bool:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
            response.raise_for_status()
            checksum = hashlib.sha256()
            with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as fh:
                tmp = Path(fh.name)
                for chunk in response.iter_bytes():
                    checksum.update(chunk)
                    fh.write(chunk)
        if checksum.hexdigest() != digest:
            raise ValueError(f"tokenizer checksum mismatch for {dest.name}")
        tmp.replace(dest)
        return True
    except httpx.HTTPError:
        return False
    finally:
        if tmp:
            tmp.unlink(missing_ok=True)


def load_vocab(spec: VocabSpec) -> Optional[Vocab]:
    """Load one vocabulary, downloading and caching it on first use.

    Returns ``None`` when the source is unavailable (gated repo, renamed model,
    no network). Unavailable candidates are skipped, never guessed at.
    """
    try:
        if spec.kind == "tiktoken":
            import tiktoken

            enc = tiktoken.get_encoding(spec.source)
            return Vocab(spec, lambda s: len(enc.encode(s, disallowed_special=())))
        if spec.kind == "hf":
            from tokenizers import Tokenizer

            dest = CACHE_DIR / "vocab" / f"{spec.key}.json"
            if not dest.is_file() or hashlib.sha256(dest.read_bytes()).hexdigest() != spec.sha256:
                url = f"https://huggingface.co/{spec.source}/resolve/{spec.revision}/{spec.filename}"
                if not _download(url, dest, spec.sha256):
                    return None
            tok = Tokenizer.from_file(str(dest))

            def counter(text: str) -> int:
                return len(tok.encode(text, add_special_tokens=False).ids)

            return Vocab(spec, counter)
    except (OSError, ImportError):
        return None
    return None


def load_all(keys: Optional[List[str]] = None) -> Dict[str, Vocab]:
    """Load every requested vocabulary, skipping unavailable ones."""
    if keys is not None and (unknown := set(keys) - _BY_KEY.keys()):
        raise ValueError("unknown vocabularies: " + ", ".join(sorted(unknown)))
    specs = [v for v in VOCABS if keys is None or v.key in keys]
    out: Dict[str, Vocab] = {}
    for spec in specs:
        vocab = load_vocab(spec)
        if vocab is not None:
            out[spec.key] = vocab
    return out


def vocab_fingerprint(vocab: Vocab) -> Dict[str, str]:
    return {"key": vocab.spec.key, "lab": vocab.spec.lab,
            "family": vocab.spec.family, "source": vocab.spec.source,
            "revision": vocab.spec.revision, "sha256": vocab.spec.sha256}
