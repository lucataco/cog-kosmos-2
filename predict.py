# Prediction interface for Cog

from cog import BasePredictor, Input, Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_NAME = "microsoft/kosmos-2-patch14-224"
MODEL_CACHE = "model-cache"
MODEL_PROC = "model-proc"

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
        )
    ) -> str:
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

        result = processed_text + "\n\n" + str(entities)
        
        return result