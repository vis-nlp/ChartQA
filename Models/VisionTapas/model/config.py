

""" VisionTapas model configuration """


from transformers import PretrainedConfig, TapasConfig, ViTConfig, LxmertConfig

class VisionTapasConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class: VisionTapas` or a


    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        tapas_config (:obj:`TapasConfig`, `optional`, defaults to None):
            Config of the TaPas Model
        vit_config (:obj:`ViTConfig`, `optional`, defaults to None):
            Config of the ViT Model
        x_layers (:obj:`int`, `optional`, defaults to 5):
            Number of hidden layers in the Transformer cross modality encoder.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        num_qa_labels (:obj:`int`, `optional`, defaults to 55):
            Number of outputs in the classification layer


    """

    model_type = "visiontapas"

    def __init__(
        self,
        x_layers=4,
        num_labels=55,
        initializer_range=0.02,
        output_attentions=False,
        output_hidden_states=False,
        **kwargs,
    ):
        super().__init__(**kwargs)


        self.x_layers = x_layers
        self.num_labels = num_labels
        self.initializer_range= initializer_range

        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.num_hidden_layers = {"cross_encoder": x_layers}


