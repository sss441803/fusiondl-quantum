from conf_parser import parameters
import os
import errno

# TODO(KGF): this conf.py feels like an unnecessary level of indirection
if os.path.exists('./quantum/conf.yaml'):
    conf = parameters('./quantum/conf.yaml')