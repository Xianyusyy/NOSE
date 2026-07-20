import json

from nose.config import load_config


def test_public_config_loads(tmp_path) -> None:
    config = {
        "format": "nose-inference-config",
        "format_version": 1,
        "base_models": {
            "qwen": "Qwen/example",
            "unimol": "dptech/Uni-Mol-Models",
            "unimol_checkpoint": "model.pt",
            "unimol_dictionary": "dict.txt",
            "esm2": "facebook/esm",
        },
        "model": {
            "text_hidden_size": 8,
            "embedding_dim": 4,
            "molecular_hidden_size": 4,
            "receptor_hidden_size": 6,
            "descriptor_adapter": {
                "d_model": 4,
                "hidden_dim": 8,
                "layers": 1,
                "dropout": 0.0,
            },
            "receptor_adapter": {
                "d_model": 4,
                "hidden_dim": 8,
                "layers": 1,
                "dropout": 0.0,
            },
        },
        "inference": {"max_text_length": 32, "hard_orthogonal": True},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.qwen_model == "Qwen/example"
    assert loaded.model.embedding_dim == 4
    assert loaded.max_text_length == 32
