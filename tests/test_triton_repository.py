from card_recognizer.serving.triton_repository import build_config_pbtxt


def test_build_config_pbtxt_for_onnx_model() -> None:
    config = build_config_pbtxt(
        model_name="card_recognizer_onnx",
        platform="onnxruntime_onnx",
        input_name="images",
        output_name="logits",
        image_size=224,
        num_classes=53,
        max_batch_size=32,
        instance_kind="KIND_CPU",
    )

    assert 'name: "card_recognizer_onnx"' in config
    assert 'platform: "onnxruntime_onnx"' in config
    assert "max_batch_size: 32" in config
    assert 'name: "images"' in config
    assert "dims: [3, 224, 224]" in config
    assert 'name: "logits"' in config
    assert "dims: [53]" in config
    assert "kind: KIND_CPU" in config


def test_build_config_pbtxt_for_tensorrt_model() -> None:
    config = build_config_pbtxt(
        model_name="card_recognizer_tensorrt",
        platform="tensorrt_plan",
        input_name="images",
        output_name="logits",
        image_size=224,
        num_classes=53,
        max_batch_size=32,
        instance_kind="KIND_GPU",
    )

    assert 'name: "card_recognizer_tensorrt"' in config
    assert 'platform: "tensorrt_plan"' in config
    assert "kind: KIND_GPU" in config
