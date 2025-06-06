from .detector3d_template import Detector3DTemplate
from .centerpoint import CenterPoint
from .sara3d import SARA3D

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'CenterPoint': CenterPoint,
    'SARA3D': SARA3D
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
