from settings import TEST

from . import light_curve_upsampling

__all__ = ['light_curve_upsampling']

if not TEST:
    from . import GP_fitting
    __all__.append('GP_fitting')
