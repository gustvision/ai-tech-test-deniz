from torch import nn
from transformers import ClapAudioModelWithProjection


class AudioClassifier(nn.Module):
    """
    Model: CLAP model is used from huggingface https://huggingface.co/docs/transformers/v4.27.2/en/model_doc/clap
    Paper: The CLAP model was proposed Large Scale Constrastive Laungaue-Audio pretraining with feature fusion and keyword-to-caption augmentation
    by Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, Shlomo Dubnov.
    Training strategy: The model has contrastive language-audio pretraining to develop an audio representation by combining audio data with natural language descriptions.
    """
    def __init__(self, model_name, dropout):
        """
        :param model_name:
        :param dropout:
        """
        super(AudioClassifier, self).__init__()

        self.backbone = ClapAudioModelWithProjection.from_pretrained(model_name)
        # TODO: make number of layers and hidden dims parametric
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def forward(self, input):
        output = self.backbone(**input).audio_embeds
        output = self.classifier(output)
        return output
