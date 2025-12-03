# -*- coding: utf-8 -*-
"""
Object detection module using SAM3 (Segment Anything Model 3).

This module provides detection capabilities using SAM3's text-guided detection.
It replaces the previous OWLv2-based detection with SAM3's more powerful
vision-language understanding.
"""

import json
import os
import sys
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch
from PIL import Image
from tqdm import tqdm

import configurations as cfg

# SAM3 imports
import sam3
from sam3 import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata,
    FindQueryLoaded,
    Image as SAMImage,
    Datapoint,
)
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.eval.postprocessors import PostProcessImage
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)

# Enable TF32 for better performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


########## Utilities ##########

def list_images(image_directory: str) -> List[str]:
    """List all image files in a directory, sorted alphabetically."""
    return_list = []
    for filename in os.listdir(image_directory):
        filepath = os.path.join(image_directory, filename)
        if os.path.isfile(filepath) and os.path.splitext(filename)[1].lower() in cfg.IMAGE_EXTENSIONS:
            return_list.append(filepath)
    return sorted(return_list)


def list_mesh_directories(mesh_root_directory: str) -> List[str]:
    """List all mesh subdirectories in the root directory."""
    return sorted([
        os.path.join(mesh_root_directory, mesh_directory)
        for mesh_directory in os.listdir(mesh_root_directory)
        if os.path.isdir(os.path.join(mesh_root_directory, mesh_directory))
    ])


def visualize_bboxes_for_keyframe(
    image_path: str,
    mesh_name: str,
    detection_output_path: str = cfg.OD_OUTPUT_PATH
) -> None:
    """Visualize detection bounding boxes for a specific keyframe and mesh."""
    import matplotlib.pyplot as plt
    from matplotlib import patches

    with open(detection_output_path, "r", encoding="utf-8") as file:
        full_results_dict = json.load(file)

    keyframe_image = Image.open(image_path).convert("RGB")
    keyframe_filename = os.path.basename(image_path)

    if mesh_name not in full_results_dict:
        print(f'Mesh "{mesh_name}" not found in detection results.', file=sys.stderr)
        return

    mesh_results_dict = full_results_dict[mesh_name]

    if keyframe_filename not in mesh_results_dict:
        print(f'No "{mesh_name}" detected in keyframe "{keyframe_filename}".', file=sys.stderr)
        return

    bboxes_list = mesh_results_dict[keyframe_filename].get("bboxes", [])
    matching_scores_list = mesh_results_dict[keyframe_filename].get("scores", [])

    if not bboxes_list or not matching_scores_list:
        print(f'Keyframe "{keyframe_filename}" has no bboxes/scores for mesh "{mesh_name}".', file=sys.stderr)
        return

    if len(bboxes_list) != len(matching_scores_list):
        print("The length of bboxes is not equal to the length of scores.", file=sys.stderr)
        return

    figure, axes = plt.subplots(1, 1)
    figure.canvas.manager.set_window_title(f"{mesh_name} @ {keyframe_filename}")
    axes.imshow(keyframe_image)
    axes.axis("off")

    for bbox, matching_score in zip(bboxes_list, matching_scores_list):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)

        rectangle_patch = patches.Rectangle(
            (x1, y1),
            box_width,
            box_height,
            linewidth=2,
            edgecolor="tab:cyan",
            facecolor="none",
        )
        axes.add_patch(rectangle_patch)

        score_label_text = f"{round(matching_score, 3)}"
        text_x = x1 + 3
        text_y = max(y1 + 12, 12)
        axes.text(
            text_x,
            text_y,
            score_label_text,
            fontsize=9,
            color="black",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
            verticalalignment="top",
        )

    plt.tight_layout()
    plt.show()


########## SAM3 Detector ##########

@dataclass
class DetectionResult:
    """Detection result for a single image, matching OWLv2 output format."""
    scores: torch.Tensor  # Shape: [N]
    boxes: torch.Tensor   # Shape: [N, 4] in XYXY format


class SAM3Detector:
    """
    SAM3-based object detector.

    This class provides an interface similar to OwlV2Detector for easy replacement.
    It uses SAM3's text-guided detection capabilities.
    """

    def __init__(
        self,
        bpe_path: Optional[str] = None,
        compile_model: bool = False,
    ):
        """
        Initialize the SAM3 detector.

        Args:
            bpe_path: Path to BPE vocabulary file. If None, uses default.
            compile_model: Whether to compile the model for faster inference.
        """
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)
        print(f"[SAM3] device={self.device}")

        # Get SAM3 root directory for default paths
        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        if bpe_path is None:
            bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

        # Build SAM3 model
        print("[SAM3] Loading model (this may take a while on first run)...")
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=self.device_name,
            eval_mode=True,
            compile=compile_model,
        )

        # Setup image transforms
        self.transform = ComposeAPI(transforms=[
            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Postprocessor will be created per-batch with the appropriate threshold
        self._postprocessor_cache = {}

        # Query ID counter for tracking results
        self._query_id_counter = 0

        print("[SAM3] Model loaded successfully.")

    def _get_postprocessor(self, detection_threshold: float) -> PostProcessImage:
        """Get or create a postprocessor with the given threshold."""
        if detection_threshold not in self._postprocessor_cache:
            self._postprocessor_cache[detection_threshold] = PostProcessImage(
                max_dets_per_img=-1,
                iou_type="segm",
                use_original_sizes_box=True,
                use_original_sizes_mask=True,
                convert_mask_to_rle=False,
                detection_threshold=detection_threshold,
                to_cpu=True,
            )
        return self._postprocessor_cache[detection_threshold]

    def _create_datapoint(
        self,
        image: Image.Image,
        query_text: str,
        image_index: int,
    ) -> tuple[Datapoint, int]:
        """
        Create a SAM3 Datapoint for a single image with a text query.

        Args:
            image: PIL Image to run detection on
            query_text: Text description of object to detect
            image_index: Index of this image in the batch (for result tracking)

        Returns:
            Tuple of (Datapoint, query_id)
        """
        w, h = image.size

        # Create datapoint with the image
        datapoint = Datapoint(
            find_queries=[],
            images=[SAMImage(data=image, objects=[], size=[h, w])],
        )

        # Create the text query
        query_id = self._query_id_counter
        self._query_id_counter += 1

        find_query = FindQueryLoaded(
            query_text=query_text,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            input_bbox=None,
            input_bbox_label=None,
            inference_metadata=InferenceMetadata(
                coco_image_id=query_id,
                original_image_id=query_id,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=image_index,
            ),
        )

        datapoint.find_queries.append(find_query)

        return datapoint, query_id

    def detect_batch_text_guided(
        self,
        prompt: str,
        target_images: List[Image.Image],
        match_score_threshold: float,
    ) -> List[DetectionResult]:
        """
        Detect objects in a batch of images using a text prompt.

        Args:
            prompt: Text description of objects to detect (e.g., "cat", "red car")
            target_images: List of PIL Images to run detection on
            match_score_threshold: Minimum confidence score for detections

        Returns:
            List of DetectionResult, one per input image.
            Each result contains:
                - scores: Tensor of confidence scores [N]
                - boxes: Tensor of bounding boxes [N, 4] in XYXY format (pixel coords)
        """
        if not target_images:
            return []

        # Reset query ID counter for this batch
        self._query_id_counter = 0

        # Create datapoints for each image
        datapoints = []
        query_id_to_image_idx = {}

        for img_idx, image in enumerate(target_images):
            datapoint, query_id = self._create_datapoint(image, prompt, img_idx)
            datapoint = self.transform(datapoint)
            datapoints.append(datapoint)
            query_id_to_image_idx[query_id] = img_idx

        # Collate into batch
        batch = collate(datapoints, dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, self.device, non_blocking=True)

        # Run inference
        with torch.inference_mode():
            with torch.amp.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                output = self.model(batch)

        # Postprocess results
        postprocessor = self._get_postprocessor(match_score_threshold)
        raw_results = postprocessor.process_results(output, batch.find_metadatas)

        # Convert to list of DetectionResult, ordered by image index
        results = []
        for img_idx in range(len(target_images)):
            # Find the query_id for this image
            query_id = None
            for qid, idx in query_id_to_image_idx.items():
                if idx == img_idx:
                    query_id = qid
                    break

            if query_id is not None and query_id in raw_results:
                result = raw_results[query_id]
                scores = result.get("scores", torch.tensor([]))
                boxes = result.get("boxes", torch.tensor([]).reshape(0, 4))

                # Ensure tensors are on CPU
                if torch.is_tensor(scores):
                    scores = scores.cpu()
                else:
                    scores = torch.tensor(scores)

                if torch.is_tensor(boxes):
                    boxes = boxes.cpu()
                else:
                    boxes = torch.tensor(boxes).reshape(-1, 4)

                results.append(DetectionResult(scores=scores, boxes=boxes))
            else:
                # No detections for this image
                results.append(DetectionResult(
                    scores=torch.tensor([]),
                    boxes=torch.tensor([]).reshape(0, 4)
                ))

        return results

    def detect_batch_image_guided(
        self,
        query_image: Image.Image,
        target_images: List[Image.Image],
        match_score_threshold: float,
        image_nms_threshold: float,
    ) -> List[DetectionResult]:
        """
        Detect objects in a batch of images using a reference image query.

        NOTE: This is a placeholder. SAM3's visual exemplar feature is not fully
        exposed in the public API. For now, this raises NotImplementedError.

        Args:
            query_image: Reference image showing the object to detect
            target_images: List of PIL Images to run detection on
            match_score_threshold: Minimum confidence score for detections
            image_nms_threshold: NMS threshold for overlapping detections

        Returns:
            List of DetectionResult, one per input image.
        """
        raise NotImplementedError(
            "Image-guided detection is not yet supported in SAM3Detector. "
            "SAM3's visual exemplar feature requires custom model configuration. "
            "Please use text-guided detection (detect_batch_text_guided) instead, "
            "or provide a text description of the object in the reference image."
        )


########## Legacy OWLv2 Detector (kept for reference) ##########

class OwlV2Detector:
    """
    OWLv2-based object detector (legacy).

    This class is kept for reference and fallback purposes.
    Consider using SAM3Detector for better performance.
    """

    def __init__(self, model_id: str = "google/owlv2-base-patch16-ensemble"):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection

        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)
        print(f"[OWLv2] device={self.device}")

        self.processor = Owlv2Processor.from_pretrained(model_id, use_fast=True)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            model_id, torch_dtype=cfg.OD_MODEL_DTYPE
        ).eval().to(self.device)
        torch.backends.cudnn.benchmark = True

    def detect_batch_image_guided(
        self,
        query_image: Image.Image,
        target_images: List[Image.Image],
        match_score_threshold: float,
        image_nms_threshold: float,
    ) -> List[Dict[str, Any]]:
        inputs = self.processor(
            images=target_images,
            query_images=query_image,
            return_tensors="pt",
            padding=True
        )
        inputs = {
            key: (value.to(self.device) if torch.is_tensor(value) else value)
            for key, value in inputs.items()
        }
        target_sizes = torch.tensor([(im.height, im.width) for im in target_images])

        with torch.inference_mode():
            with torch.amp.autocast(device_type=self.device_name, dtype=cfg.OD_MODEL_DTYPE):
                outputs = self.model.image_guided_detection(**inputs)

        results = self.processor.post_process_image_guided_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=match_score_threshold,
            nms_threshold=image_nms_threshold
        )
        return results

    def detect_batch_text_guided(
        self,
        prompt: str,
        target_images: List[Image.Image],
        match_score_threshold: float,
    ) -> List[Dict[str, Any]]:
        inputs = self.processor(
            text=[prompt for _ in range(len(target_images))],
            images=target_images,
            return_tensors="pt",
            padding=True
        )
        inputs = {
            key: (value.to(self.device) if torch.is_tensor(value) else value)
            for key, value in inputs.items()
        }
        target_sizes = torch.tensor([(im.height, im.width) for im in target_images])

        with torch.inference_mode():
            with torch.amp.autocast(device_type=self.device_name, dtype=cfg.OD_MODEL_DTYPE):
                outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=match_score_threshold
        )
        return results


########## Detection Pipeline ##########

def process_mesh(
    query_mode: str,
    detector: SAM3Detector,
    mesh_directory: str,
    keyframe_paths: List[str],
    query_text_filename: str,
    batch_size: int,
    match_score_threshold: float,
    minimum_box_area: int,
    image_nms_threshold: float,  # Not used for SAM3 text-guided, kept for API compatibility
) -> Dict[str, Dict[str, Any]]:
    """
    Process a single mesh directory and detect objects in all keyframes.

    Args:
        query_mode: "text" for text-guided detection, "image" for image-guided
        detector: SAM3Detector instance
        mesh_directory: Path to mesh directory containing query text/image
        keyframe_paths: List of keyframe image paths to process
        query_text_filename: Filename of text query file in mesh directory
        batch_size: Number of images to process per batch
        match_score_threshold: Minimum confidence score for detections
        minimum_box_area: Minimum bounding box area in pixels
        image_nms_threshold: NMS threshold (not used for SAM3 text-guided)

    Returns:
        Dictionary mapping keyframe filenames to detection results
    """
    # Load query
    query_image = None
    query_text = None

    if query_mode == "image":
        # For image-guided mode, we need to fall back to text if available
        # since SAM3 doesn't support image-guided detection yet
        query_text_path = os.path.join(mesh_directory, query_text_filename)
        if os.path.exists(query_text_path):
            with open(query_text_path, "r", encoding="utf-8") as f:
                query_text = f.read().strip()
            print(f"[SAM3] Image-guided mode requested but using text fallback: '{query_text}'")
        else:
            # Try to use mesh directory name as query text
            query_text = os.path.basename(mesh_directory)
            print(f"[SAM3] No text file found, using directory name as query: '{query_text}'")
    else:
        # Text-guided mode
        query_text_path = os.path.join(mesh_directory, query_text_filename)
        with open(query_text_path, "r", encoding="utf-8") as f:
            query_text = f.read().strip()

    detection_results = {}
    keyframe_paths = copy.deepcopy(keyframe_paths)

    with tqdm(total=len(keyframe_paths), desc=os.path.basename(mesh_directory), unit="keyframe") as pbar:
        while keyframe_paths:
            # Build batch
            batch_info = []
            batch_images = []

            while len(batch_info) < batch_size and keyframe_paths:
                keyframe_path = keyframe_paths.pop(0)
                try:
                    keyframe_image = Image.open(keyframe_path).convert("RGB")
                    batch_info.append({
                        "path": keyframe_path,
                        "size": (keyframe_image.height, keyframe_image.width),
                    })
                    batch_images.append(keyframe_image)
                except Exception as e:
                    print(f"Failed to load {keyframe_path}: {e}", file=sys.stderr)

            if not batch_images:
                break

            # Run detection
            try:
                results = detector.detect_batch_text_guided(
                    prompt=query_text,
                    target_images=batch_images,
                    match_score_threshold=match_score_threshold,
                )
            except Exception as e:
                print(f"Detection failed for batch: {e}", file=sys.stderr)
                results = [DetectionResult(scores=torch.tensor([]), boxes=torch.tensor([]).reshape(0, 4))
                          for _ in batch_images]

            # Process results
            for idx, result in enumerate(results):
                keyframe_path = batch_info[idx]["path"]
                keyframe_height, keyframe_width = batch_info[idx]["size"]

                filtered_bboxes = []
                filtered_scores = []

                scores = result.scores.tolist() if torch.is_tensor(result.scores) else result.scores
                boxes = result.boxes.tolist() if torch.is_tensor(result.boxes) else result.boxes

                for score, box in zip(scores, boxes):
                    x1, y1, x2, y2 = box

                    # Clamp to image bounds
                    x1c = round(max(0, min(x1, keyframe_width - 1)))
                    y1c = round(max(0, min(y1, keyframe_height - 1)))
                    x2c = round(max(0, min(x2, keyframe_width - 1)))
                    y2c = round(max(0, min(y2, keyframe_height - 1)))

                    # Filter by minimum area
                    area = max(0, x2c - x1c) * max(0, y2c - y1c)
                    if area >= minimum_box_area:
                        filtered_bboxes.append([x1c, y1c, x2c, y2c])
                        filtered_scores.append(float(score))

                if filtered_bboxes:
                    detection_results[os.path.basename(keyframe_path)] = {
                        "bboxes": filtered_bboxes,
                        "scores": filtered_scores,
                    }

            pbar.update(len(batch_info))

            # Cleanup
            for img in batch_images:
                try:
                    img.close()
                except Exception:
                    pass

            torch.cuda.empty_cache()

    return detection_results


def detect_meshes(
    keyframe_directory: str = cfg.OD_KEYFRAME_DIRECTORY,
    mesh_root_directory: str = cfg.OD_MESH_ROOT_DIRECTORY,
    output_path: str = cfg.OD_OUTPUT_PATH,
    query_mode: str = cfg.OD_QUERY_MODE,
    query_text_filename: str = cfg.OD_QUERY_TEXT_FILENAME,
    batch_size: int = cfg.OD_BATCH_SIZE,
    match_score_threshold: float = cfg.OD_MATCH_SCORE_THRESHOLD,
    minimum_box_area: int = cfg.OD_MINIMUM_BOX_AREA,
    image_nms_threshold: float = cfg.OD_IMAGE_NMS_THRESHOLD,
) -> None:
    """
    Run object detection on all meshes across all keyframes.

    Args:
        keyframe_directory: Directory containing keyframe images
        mesh_root_directory: Root directory containing mesh subdirectories
        output_path: Path to save detection results JSON
        query_mode: "text" or "image" (image falls back to text for SAM3)
        query_text_filename: Name of text query file in each mesh directory
        batch_size: Number of images to process per batch
        match_score_threshold: Minimum confidence score for detections
        minimum_box_area: Minimum bounding box area in pixels
        image_nms_threshold: NMS threshold (not used for SAM3 text-guided)
    """
    keyframe_paths = list_images(keyframe_directory)
    mesh_directories = list_mesh_directories(mesh_root_directory)

    print(f"Found {len(keyframe_paths)} keyframes and {len(mesh_directories)} meshes.")

    # Initialize SAM3 detector
    detector = SAM3Detector()

    final_results = {}
    for mesh_directory in mesh_directories:
        detection_results = process_mesh(
            query_mode=query_mode,
            detector=detector,
            mesh_directory=mesh_directory,
            keyframe_paths=keyframe_paths,
            query_text_filename=query_text_filename,
            batch_size=batch_size,
            match_score_threshold=match_score_threshold,
            minimum_box_area=minimum_box_area,
            image_nms_threshold=image_nms_threshold,
        )
        final_results[os.path.basename(mesh_directory)] = detection_results

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)

    print(f"Saved detection results to {output_path}.")


if __name__ == "__main__":
    # Run detection with text queries
    detect_meshes(query_mode="text")

    # Example visualization (uncomment to use):
    # visualize_bboxes_for_keyframe(
    #     os.path.join(cfg.OD_KEYFRAME_DIRECTORY, "keyframe_435.jpg"),
    #     "bear"
    # )
