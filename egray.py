#!/usr/bin/python

"""egray.py

Usage:
    ./egray.py \
[-d <decay> | --decay=<decay>] \
[-n <noise> | --noise=<noise>] \
[-s <step> | --step=<step>] \
[-X <width> | --width=<width>] \
[-Y <height> | --height=<height>]
    ./egray.py (-h | --help)
    ./egray.py (--version)

Options:
    -d <decay>, --decay <decay>     # [default: 3e-1]
    -n <noise>, --noise <noise>     # [default: 3e-1]
    -s <step>, --step <step>        # [default: 3e-2]
    -X <width>, --width <width>     # [default: 512]
    -Y <height>, --height <height>  # [default: 512]
    -h, --help
    --version
"""

from scipy import (ones, exp)
from scipy.misc import (toimage)
from Image import (fromarray)
from scipy import random
from docopt import (docopt)

def make(**kw):
    step, noise, decay = [float(kw[k]) for k in ['--step','--noise','--decay']]
    X, Y = [int(kw[k]) for k in ['--width', '--height']]
    H = X/2

    gray = ones((Y,X),dtype=float)
    for x in range(H):
        dI = step * exp(-decay*float(H-x))
        gray[:,0+x+0] += dI
        gray[:,X-x-1] -= dI

    scatter = ((random.random((Y,X))*2)-1) * noise
    image = (gray+scatter)*127

    saved = fromarray(image)
    name = 'egray/egray.s.%f.n.%f.d.%f.gif' % (step, noise, decay)
    saved.save(name)
    toimage(saved)

if __name__ == "__main__":
    kwargs = docopt(__doc__, version="0.0.1")
    make(**kwargs)
