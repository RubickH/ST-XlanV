from models.xlanv import XLANV
from models.xlanv_ens import XLANV_ENS

__factory = {
    'XLANV': XLANV,
    'XLANV_ENS': XLANV_ENS
}

def names():
    return sorted(__factory.keys())

def create(name, pos=True, lang=True):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](pos, lang)