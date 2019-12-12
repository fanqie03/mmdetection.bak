from .registry import DATASETS
from .xml_style import XMLDataset
@DATASETS.register_module
class HelmetDataset(XMLDataset):
    CLASSES = ('hat', 'person')
    def __init__(self, **kwargs):
        super(HelmetDataset, self).__init__(**kwargs)

