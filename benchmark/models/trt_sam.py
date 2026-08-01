"""TensorRT predictor for EfficientViT-SAM.

EfficientViT ships ONNX export scripts but no TensorRT inference wrapper (its
``deployment/`` directory contains only ``onnx/``), so this is the piece that
has to exist here rather than upstream. NanoSAM needs no equivalent -- it
ships its own ``nanosam.utils.predictor.Predictor``.

The interface deliberately mirrors ``EfficientViTSamPredictor``
(``set_image`` / ``predict``) so ``EfficientViTSamSegmenter`` can swap between
the two runtimes without caring which is live, and -- more importantly for an
ablation study -- so both runtimes produce the *same* masks. Everything that
is not the encoder/decoder forward pass is therefore copied from upstream's
own pre/postprocessing rather than reimplemented: same resize, same
normalisation, same padding, same mask upscaling, same mask selection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

#: ``EfficientViTSam.image_size``, per model, from ``build_efficientvit_sam()``
#: -- ``image_size=(1024, image_size)``, where the l-series defaults to 512 and
#: the xl-series to 1024.
#:
#: The two entries mean different things and are easy to conflate. [1] is the
#: square the *encoder* consumes (the frame is resized so its long side hits
#: this, then corner-padded). [0] is the reference frame *prompt coordinates*
#: live in -- the exported decoder divides point coords by it -- and the
#: intermediate size masks are upscaled to before cropping. They are only
#: equal for the xl models.
_IMAGE_SIZES: dict[str, tuple[int, int]] = {
    "efficientvit-sam-l0": (1024, 512),
    "efficientvit-sam-l1": (1024, 512),
    "efficientvit-sam-l2": (1024, 512),
    "efficientvit-sam-xl0": (1024, 1024),
    "efficientvit-sam-xl1": (1024, 1024),
}


def _load_engine(path: str, input_names: list[str], output_names: list[str]) -> Any:
    import tensorrt as trt
    from torch2trt import TRTModule

    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
    return TRTModule(engine=engine, input_names=input_names, output_names=output_names)


def _build_transform(encoder_side: int):
    """Rebuild ``EfficientViTSam.transform`` without instantiating the model.

    Constructing the real module to borrow its transform would allocate the
    whole backbone -- the thing the engines exist to replace. The two
    components that carry logic (``SamResize``, ``SamPad``) are imported from
    upstream so they cannot drift; only the normalisation constants are
    repeated, and a mismatch there shows up immediately as garbage masks.
    """
    import torchvision.transforms as transforms

    from efficientvit.models.efficientvit.sam import SamPad, SamResize

    return transforms.Compose(
        [
            SamResize(encoder_side),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
            ),
            SamPad(encoder_side),
        ]
    )


class TrtSamPredictor:
    #: ``EfficientViTSam.mask_threshold``.
    mask_threshold: float = 0.0

    def __init__(
        self,
        encoder_engine: str,
        decoder_engine: str,
        model: str = "efficientvit-sam-l0",
        device: str = "cuda",
    ) -> None:
        try:
            self.image_size = _IMAGE_SIZES[model]
        except KeyError:
            raise ValueError(
                f"Unknown EfficientViT-SAM model {model!r}. "
                f"Known: {', '.join(sorted(_IMAGE_SIZES))}"
            ) from None

        self.device = device
        # Names come from the exported graphs -- see the export scripts in
        # efficientvit's applications/efficientvit_sam/deployment/onnx/.
        self.encoder = _load_engine(encoder_engine, ["input_image"], ["image_embeddings"])
        self.decoder = _load_engine(
            decoder_engine,
            ["image_embeddings", "point_coords", "point_labels"],
            ["masks", "iou_predictions"],
        )
        self._transform = _build_transform(self.image_size[1])
        self.reset_image()

    def reset_image(self) -> None:
        self.is_image_set = False
        self.features: torch.Tensor | None = None
        self.original_size: tuple[int, int] | None = None
        self.input_size: tuple[int, int] | None = None

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """Scale (X, Y) pixels from the source frame into the prompt frame."""
        old_h, old_w = self.original_size  # type: ignore[misc]
        new_h, new_w = self.input_size  # type: ignore[misc]
        coords = coords.astype(float).copy()
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray) -> np.ndarray:
        return self.apply_coords(boxes.reshape(-1, 2, 2)).reshape(-1, 4)

    @torch.inference_mode()
    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        from segment_anything.utils.transforms import ResizeLongestSide

        if image_format not in ("RGB", "BGR"):
            raise ValueError(f"image_format must be RGB or BGR, got {image_format!r}")
        if image_format != "RGB":
            image = image[..., ::-1]

        self.reset_image()
        self.original_size = image.shape[:2]
        self.input_size = ResizeLongestSide.get_preprocess_shape(
            *self.original_size, long_side_length=self.image_size[0]
        )
        tensor = self._transform(image).unsqueeze(dim=0).to(self.device)
        self.features = self.encoder(tensor)
        self.is_image_set = True

    def _postprocess_masks(self, masks: torch.Tensor) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_size[0], self.image_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : self.input_size[0], : self.input_size[1]]  # type: ignore[index]
        return F.interpolate(masks, self.original_size, mode="bilinear", align_corners=False)

    @torch.inference_mode()
    def predict(
        self,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        mask_input: np.ndarray | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before predicting.")
        if mask_input is not None:
            # The exported decoder takes no mask_input; upstream's export
            # drops it and always uses the no_mask embedding.
            raise NotImplementedError(
                "mask_input is not supported by the exported EfficientViT-SAM decoder."
            )

        coords: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        if point_coords is not None:
            if point_labels is None:
                raise ValueError("point_labels must be supplied with point_coords.")
            coords.append(self.apply_coords(np.asarray(point_coords, dtype=float)))
            labels.append(np.asarray(point_labels, dtype=float))
        if box is not None:
            box_arr = self.apply_boxes(np.asarray(box, dtype=float).reshape(1, 4))
            coords.append(box_arr.reshape(2, 2))
            # 2 = box top-left, 3 = box bottom-right. The exported decoder
            # has no `boxes` input, so a box is passed as its two corners
            # under SAM's corner labels -- which index the same
            # point_embeddings the PyTorch prompt encoder uses for boxes.
            labels.append(np.array([2.0, 3.0]))
        if not coords:
            raise ValueError("predict() needs point_coords or box.")

        coords_t = torch.from_numpy(
            np.concatenate(coords, axis=0)[None].astype(np.float32)
        ).to(self.device)
        labels_t = torch.from_numpy(
            np.concatenate(labels, axis=0)[None].astype(np.float32)
        ).to(self.device)

        masks, iou_predictions = self.decoder(self.features, coords_t, labels_t)

        # The engine is built from an export without --return-single-mask, so
        # it returns all four mask tokens and this slice reproduces
        # MaskDecoder.forward's own: token 0 is the single-mask output, 1..3
        # the multimask ones. Selecting here rather than in the graph is what
        # keeps the TensorRT and PyTorch runtimes returning the same mask --
        # upstream's --return-single-mask instead picks argmax over all four,
        # which is a different answer.
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        low_res_masks = masks[:, mask_slice]
        iou_predictions = iou_predictions[:, mask_slice]

        full_masks = self._postprocess_masks(low_res_masks)
        if not return_logits:
            full_masks = full_masks > self.mask_threshold

        return (
            full_masks[0].detach().cpu().numpy(),
            iou_predictions[0].detach().cpu().numpy(),
            low_res_masks[0].detach().cpu().numpy(),
        )
