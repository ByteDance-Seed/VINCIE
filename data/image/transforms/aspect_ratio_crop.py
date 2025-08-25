 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

from typing import List
import torch


class AspectRatioCrop:
    def __init__(
        self,
        aspect_ratio: str,
        adaptive_aspect_ratio_cands: List[str] = ["9:16", "1:1", "16:9"],
    ):
        """
        aspect_ratio (witdh:height):
            'keep_ratio': keep the ratio of original image
            ['16:9','4:3','1:1','3:4','9:16']: center crop the image to the target ratio
            'adaptive': adaptive choose the target ratio from adaptive_aspect_ratio_cands
        """
        self.aspect_ratio = aspect_ratio
        self.adaptive_aspect_ratio_cands = adaptive_aspect_ratio_cands

    def _get_aspect_ratio_coordinates(self, original_height, original_width, aspect_ratio):
        aspect_ratio = [float(r) for r in aspect_ratio.split(":")]
        assert len(aspect_ratio) == 2, "aspect_ratio must be a list of two numbers"
        if original_height / original_width >= aspect_ratio[1] / aspect_ratio[0]:
            delta_h = (original_height - original_width * aspect_ratio[1] / aspect_ratio[0]) / 2
            h1, h2 = int(delta_h), int(original_height - delta_h)
            w1, w2 = 0, original_width
        else:
            delta_w = (original_width - original_height / aspect_ratio[1] * aspect_ratio[0]) / 2
            h1, h2 = 0, original_height
            w1, w2 = int(delta_w), int(original_width - delta_w)
        return h1, h2, w1, w2

    def __call__(self, image: torch.Tensor):
        original_height, original_width = image.shape[-2:]
        if self.aspect_ratio == "keep_ratio":
            return image
        if self.aspect_ratio == "adaptive":
            max_cropped_area = -1
            for ratio in self.adaptive_aspect_ratio_cands:
                h1, h2, w1, w2 = self._get_aspect_ratio_coordinates(
                    original_height, original_width, ratio
                )
                new_area = (w2 - w1) * (h2 - h1)
                if new_area > max_cropped_area:
                    max_cropped_area = new_area
                    best_h1, best_h2, best_w1, best_w2 = h1, h2, w1, w2
            image = image[:, best_h1:best_h2, best_w1:best_w2]
            return image
        assert self.aspect_ratio in ["16:9", "4:3", "1:1", "3:4", "9:16"]
        h1, h2, w1, w2 = self._get_aspect_ratio_coordinates(
            original_height, original_width, self.aspect_ratio
        )
        image = image[:, h1:h2, w1:w2]
        return image
