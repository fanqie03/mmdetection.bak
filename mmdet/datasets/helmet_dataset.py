from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class HelmetDataset(XMLDataset):
    """
    https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset
    """
    CLASSES = ('hat', 'person', 'dog')
    def __init__(self, **kwargs):
        super(HelmetDataset, self).__init__(**kwargs)
        self.cat2label = {'hat': 1, 'person': 2, 'dog': 1}  # dog是戴上的
        print(self.cat2label)


@DATASETS.register_module
class HelmetDatasetP2(XMLDataset):
    """
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/7CBGOS
    """
    CLASSES = ('white', 'yellow', 'red', 'blue', 'none')
    def __init__(self, **kwargs):
        super(HelmetDatasetP2, self).__init__(**kwargs)
        self.cat2label = {'white': 1, 'yellow': 1, 'red': 1, 'blue': 1, 'none': 2}
        print(self.cat2label)

@DATASETS.register_module
class HelmetDatasetP3(XMLDataset):
    """
    https://pythonawesome.com/helmet-detection-on-construction-sites/
    """
    CLASSES = ('helmet', 'head', 'person')

    def __init__(self, **kwargs):
        super(HelmetDatasetP3, self).__init__(**kwargs)
        self.cat2label = {'helmet': 1, 'head': 2}  # 忽略person类别
        print(self.cat2label)


@DATASETS.register_module
class HelmetMergeDataset(XMLDataset):
    """
    https://pythonawesome.com/helmet-detection-on-construction-sites/
    """
    CLASSES = ('helmet', 'head')

    def __init__(self, **kwargs):
        super(HelmetMergeDataset, self).__init__(**kwargs)
        self.cat2label = {'helmet': 1, 'head': 2}  # 忽略person类别
        print(self.cat2label)