from transformers import AutoConfig


class Meta:
    config: AutoConfig

    def __init__(self, modelName, tritonUrl):
		 self.model_name = modelName
		 self.tritonUrl = tritonUrl

    def get(self):
		 return {
			 'model_name': self.modelName,
			 'triton_url': self.tritonUrl,
		 }
