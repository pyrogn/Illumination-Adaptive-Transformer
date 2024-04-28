import base64
import io
import os
import torch
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
from IAT_enhance.model.IAT import IAT


class IATHandler(BaseHandler):
    """
    A custom model handler implementation for the IAT model.
    """

    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.initialized = False

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model.
        :param context: context contains model server system properties.
        """
        # Load the model
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        # Check if the model file exists
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        # Load the model

        # it should be implicit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = IAT().to(self.device)
        self.model.load_state_dict(torch.load(model_pt_path, map_location=self.device))
        self.model.eval()

        self.initialized = True

    def preprocess(self, data):
        """
        Preprocessing involves transforming the input image to a tensor.
        :param data: Input data for prediction.
        :return: Preprocessed input data.
        """
        images = []

        for row in data:
            # Convert the input image to a tensor
            image = row.get("data") or row.get("body")
            image = Image.open(io.BytesIO(image))
            image = self.transform(image)
            images.append(image)

        return torch.stack(images).to(self.device)

    def inference(self, inputs):
        """
        Run model prediction on the preprocessed data.
        :param inputs: Preprocessed input data.
        :return: Model prediction output.
        """
        # Forward pass through the model
        with torch.no_grad():
            _, _, enhanced_img = self.model(inputs)

        return enhanced_img

    def postprocess(self, inference_output):
        """
        Postprocessing involves converting the model output tensors to base64-encoded strings.
        :param inference_output: Inference output.
        :return: Postprocessed inference output.
        """
        output_data = []
        for tensor in inference_output:
            image = transforms.ToPILImage()(tensor.cpu().squeeze(0))
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            base64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
            output_data.append(base64_img)

        return output_data
