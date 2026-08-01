#!/usr/bin/env python3
"""Export EfficientViT-SAM's image encoder and mask decoder to ONNX.

This exists instead of calling efficientvit's own export scripts directly
because those declare ``point_coords``/``point_labels`` with a dynamic batch
axis, and SAM's mask decoder runs

    torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)

With the batch symbolic, that repeat count is symbolic too, and torch lowers
it to ``OneHot`` -> ``Tile`` where the OneHot result feeds Tile's *repeats*
input -- a shape tensor. TensorRT rejects precisely that:

    Error Code 4: Internal Error (/OneHot: an IIOneHotLayer cannot be used
    to compute a shape tensor)

which is the same wall NanoSAM's mask decoder hits (NVIDIA-AI-IOT/nanosam#16).
Measured on the real graph: pinning the batch axis alone does *not* help --
the repeat count stays symbolic and both OneHot nodes survive. Exporting
fully static folds them away entirely (the decoder drops from 1075 nodes to
436, OneHot 2 -> 0).

Static shapes cost nothing here. This app segments one detection box at a
time, and a box prompt is exactly one batch of two points -- its corners,
under SAM's corner labels 2 and 3. Fixing the shape also removes TensorRT's
dynamic-shape bookkeeping from the measured decode stage, which is a bonus
for a latency benchmark.

Everything that shapes the graph (``EncoderOnnxModel``, ``DecoderOnnxModel``)
is imported from the efficientvit clone rather than reimplemented here, so
what gets traced stays upstream's model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import warnings

REL_EXPORT_DIR = os.path.join("applications", "efficientvit_sam", "deployment", "onnx")


def find_export_dir(explicit: str | None) -> str:
    """Locate the clone's export scripts.

    They live under ``applications/``, which is not part of the installed
    ``efficientvit`` package -- so only a git clone has them, wherever it
    happens to be.
    """
    if explicit:
        if os.path.isfile(os.path.join(explicit, "export_encoder.py")):
            return explicit
        raise SystemExit(f"No export_encoder.py under {explicit}")

    candidates = [os.path.join(os.environ.get("SRC_DIR", os.getcwd()), "efficientvit")]
    try:
        from importlib.metadata import distribution

        dist = distribution("efficientvit")
        # An editable install records the source tree it points at. A regular
        # install doesn't, and locate_file() lands in site-packages, where
        # applications/ won't exist -- the isfile check below rejects it.
        raw = dist.read_text("direct_url.json")
        if raw:
            url = json.loads(raw).get("url", "")
            if url.startswith("file://"):
                candidates.append(url[len("file://") :])
        candidates.append(str(dist.locate_file("")))
    except Exception:
        pass

    for root in candidates:
        path = os.path.join(root, REL_EXPORT_DIR)
        if root and os.path.isfile(os.path.join(path, "export_encoder.py")):
            return path

    raise SystemExit(
        "Cannot find efficientvit's ONNX export scripts.\n\n"
        "They live in the git clone, under\n"
        f"  {REL_EXPORT_DIR}/\n"
        "and are not part of the installed package, so a clone must be on disk:\n\n"
        "    git clone https://github.com/mit-han-lab/efficientvit\n"
        "    pip install -e ./efficientvit --no-deps\n\n"
        "Pass --export-dir, or set SRC_DIR to the directory holding the clone."
    )


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export(model, args: tuple, output: str, input_names, output_names) -> None:
    import torch

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    kwargs = dict(
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # torch >= 2.9 defaults to the dynamo exporter, which fails on
            # this graph ("Invalid ranges [16:1]"). The TorchScript exporter
            # is what produces the graph validated here.
            torch.onnx.export(model, args, output, dynamo=False, **kwargs)
        except TypeError:
            # torch < 2.6 has no dynamo kwarg and no dynamo exporter.
            torch.onnx.export(model, args, output, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="efficientvit-sam-l0")
    parser.add_argument("--weights", required=True, help="Local .pt checkpoint.")
    parser.add_argument("--encoder-output", required=True)
    parser.add_argument("--decoder-output", required=True)
    parser.add_argument("--export-dir", default=None, help="efficientvit's deployment/onnx dir.")
    args = parser.parse_args()

    export_dir = find_export_dir(args.export_dir)
    # The clone's own root has to be importable for its export scripts to
    # resolve `from efficientvit...` when efficientvit isn't pip-installed.
    sys.path.insert(0, os.path.abspath(os.path.join(export_dir, *([os.pardir] * 4))))

    import torch  # noqa: F401  (imported after sys.path is set up)

    from efficientvit.sam_model_zoo import create_efficientvit_sam_model

    encoder_mod = load_module("evit_export_encoder", os.path.join(export_dir, "export_encoder.py"))
    decoder_mod = load_module("evit_export_decoder", os.path.join(export_dir, "export_decoder.py"))

    print(f"Loading {args.model} from {args.weights} ...")
    sam = create_efficientvit_sam_model(args.model, True, args.weights).eval()

    if not os.path.exists(args.encoder_output):
        side = sam.image_size[1]
        print(f"Exporting image encoder ({side}x{side}) -> {args.encoder_output}")
        export(
            encoder_mod.EncoderOnnxModel(model=sam),
            (torch.randn(1, 3, side, side),),
            args.encoder_output,
            ["input_image"],
            ["image_embeddings"],
        )
    else:
        print(f"{args.encoder_output} already present, skipping")

    if not os.path.exists(args.decoder_output):
        embed_dim = sam.prompt_encoder.embed_dim
        embed_size = sam.prompt_encoder.image_embedding_size
        print(f"Exporting mask decoder (1 box = 2 points) -> {args.decoder_output}")
        # return_single_mask=False deliberately: that flag makes the graph
        # pick argmax over all four mask tokens, which is NOT what the
        # PyTorch path does for multimask_output=False (it takes token 0).
        # Exporting all four and slicing in TrtSamPredictor keeps the
        # TensorRT and PyTorch runtimes returning identical masks.
        export(
            decoder_mod.DecoderOnnxModel(model=sam, return_single_mask=False),
            (
                torch.randn(1, embed_dim, *embed_size),
                torch.randint(low=0, high=1024, size=(1, 2, 2)).float(),
                torch.randint(low=0, high=4, size=(1, 2)).float(),
            ),
            args.decoder_output,
            ["image_embeddings", "point_coords", "point_labels"],
            ["masks", "iou_predictions"],
        )
    else:
        print(f"{args.decoder_output} already present, skipping")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
