# Prediction interface for Cog

from cog import BasePredictor, BaseModel, Input, Path
import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Optional
import torchvision.transforms as T
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_NAME = "microsoft/kosmos-2-patch14-224"
MODEL_CACHE = "model-cache"
MODEL_PROC = "model-proc"

colors = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (114, 128, 250),
    (0, 165, 255),
    (0, 128, 0),
    (144, 238, 144),
    (238, 238, 175),
    (255, 191, 0),
    (0, 128, 0),
    (226, 43, 138),
    (255, 0, 255),
    (0, 215, 255),
    (255, 0, 0),
]

color_map = {
    f"{color_id}": f"#{hex(color[2])[2:].zfill(2)}{hex(color[1])[2:].zfill(2)}{hex(color[0])[2:].zfill(2)}" for color_id, color in enumerate(colors)
}

def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)

class Output(BaseModel):
    img: Optional[Path]
    text: str

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE
        ).to('cuda')
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_PROC
        )
    
    def draw_entity_boxes_on_image(self, image, entities, show=False, save_path=None, entity_index=-1):
        if isinstance(image, Image.Image):
            image_h = image.height
            image_w = image.width
            image = np.array(image)[:, :, [2, 1, 0]]
        elif isinstance(image, str):
            if os.path.exists(image):
                pil_img = Image.open(image).convert("RGB")
                image = np.array(pil_img)[:, :, [2, 1, 0]]
                image_h = pil_img.height
                image_w = pil_img.width
            else:
                raise ValueError(f"invaild image path, {image}")
        elif isinstance(image, torch.Tensor):
            # pdb.set_trace()
            image_tensor = image.cpu()
            reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
            reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
            image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
            pil_img = T.ToPILImage()(image_tensor)
            image_h = pil_img.height
            image_w = pil_img.width
            image = np.array(pil_img)[:, :, [2, 1, 0]]
        else:
            raise ValueError(f"invaild image format, {type(image)} for {image}")

        if len(entities) == 0:
            return image

        indices = list(range(len(entities)))
        if entity_index >= 0:
            indices = [entity_index]

        entities = entities[:len(color_map)]

        new_image = image.copy()
        previous_bboxes = []
        text_size = 1
        text_line = 1
        box_line = 3
        (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
        base_height = int(text_height * 0.675)
        text_offset_original = text_height - base_height
        text_spaces = 3

        used_colors = colors  # random.sample(colors, k=num_bboxes)

        color_id = -1
        for entity_idx, (entity_name, (start, end), bboxes) in enumerate(entities):
            color_id += 1
            if entity_idx not in indices:
                continue
            for bbox_id, (x1_norm, y1_norm, x2_norm, y2_norm) in enumerate(bboxes):
                orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)
                color = used_colors[color_id]
                new_image = cv2.rectangle(new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line)

                l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1

                x1 = orig_x1 - l_o
                y1 = orig_y1 - l_o

                if y1 < text_height + text_offset_original + 2 * text_spaces:
                    y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
                    x1 = orig_x1 + r_o

                # add text background
                (text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
                text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - (text_height + text_offset_original + 2 * text_spaces), x1 + text_width, y1

                for prev_bbox in previous_bboxes:
                    while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox):
                        text_bg_y1 += (text_height + text_offset_original + 2 * text_spaces)
                        text_bg_y2 += (text_height + text_offset_original + 2 * text_spaces)
                        y1 += (text_height + text_offset_original + 2 * text_spaces)

                        if text_bg_y2 >= image_h:
                            text_bg_y1 = max(0, image_h - (text_height + text_offset_original + 2 * text_spaces))
                            text_bg_y2 = image_h
                            y1 = image_h
                            break

                alpha = 0.5
                for i in range(text_bg_y1, text_bg_y2):
                    for j in range(text_bg_x1, text_bg_x2):
                        if i < image_h and j < image_w:
                            if j < text_bg_x1 + 1.35 * c_width:
                                # original color
                                bg_color = color
                            else:
                                # white
                                bg_color = [255, 255, 255]
                            new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(np.uint8)

                cv2.putText(
                    new_image, f"  {entity_name}", (x1, y1 - text_offset_original - 1 * text_spaces), cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA
                )
                # previous_locations.append((x1, y1))
                previous_bboxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))
        
        pil_image = Image.fromarray(new_image[:, :, [2, 1, 0]])
        if save_path:
            pil_image.save(save_path)
        if show:
            pil_image.show()

        return pil_image

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Input image"),
        description_type: str = Input(
            description="Description Type",
            default="Brief",
            choices=[
                "Brief",
                "Detailed"
            ],
        ),
        visual_output: bool = Input(
            description="Select to show the image with bounding boxes",
            default=True
        ),
    ) -> Output:
        """Run a single prediction on the model"""
        prompt = "<grounding>An image of"
        if description_type == "Detailed":
            prompt = "<grounding>Describe this image in detail:"

        image_input = Image.open(image).convert('RGB')
        inputs = self.processor(text=prompt, images=image_input, return_tensors="pt").to('cuda')
        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        processed_text, entities = self.processor.post_process_generation(generated_text)
        result_txt = processed_text + "\n\n" + str(entities)

        if visual_output:
            annotated_image = self.draw_entity_boxes_on_image(image_input, entities)
            output_path = "/tmp/output.jpg"
            annotated_image.save(output_path)
            return Output(img=Path(output_path), text=result_txt)
        
        return Output(text=result_txt)