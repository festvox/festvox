import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
from models import *
from blocks import *


class ValenceSeq2Seq(nn.Module):

    def __init__(self):
        super(ValenceSeq2Seq, self).__init__()
        self.encoder = Encoder_TacotronOne(80)
        self.mel2output = nn.Linear(256, 3)

    def forward(self, mel):
        mel = self.encoder(mel)
        val_prediction = self.mel2output(mel)
        return val_prediction[:,-1]

