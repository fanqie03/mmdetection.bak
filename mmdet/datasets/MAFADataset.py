from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class MAFADataset(XMLDataset):

    CLASSES = ('simple', 'complex', 'human_body', 'Unspecified', )

    def __init__(self, **kwargs):
        super(MAFADataset, self).__init__(**kwargs)

@DATASETS.register_module
class MAFADatasetV2(XMLDataset):

    CLASSES = ('simple', 'complex', 'face', )

    def __init__(self, **kwargs):
        super(MAFADatasetV2, self).__init__(**kwargs)