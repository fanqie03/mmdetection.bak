from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class PikachuDataset(XMLDataset):

    CLASSES = ('pikachu',)

    def __init__(self, **kwargs):
        super(PikachuDataset, self).__init__(**kwargs)