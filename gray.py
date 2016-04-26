#!/usr/bin/python

#from scipy.misc import (toimage)
from Image import (fromarray)
from scipy import (random, ones)

def make(**kw):
    step  = kw.get( 'step', 3e-2)
    scale = kw.get('scale', 3e-1)
    shape = (Y, X) = [kw.get(edge, 512) for edge in ['Y', 'X']]
    X0, X1, X2, X3 = 0, X/2-1, X/2, X-1

    gray = ones(shape, dtype=float)
    gray[:,X0:X1] -= step
    gray[:,X2:X3] += step

    noise = ((random.random((Y,X))*2)-1) * scale

    image = (gray+noise)*127
    print gray.max(), noise.max(), image.max()
    print gray.min(), noise.min(), image.min()

    saved = fromarray(image)
    name = 'gray/gray.step.%f.scale.%f.gif' % (step, scale)
    print name
    saved.save(name)
    #toimage(saved)

if __name__ == "__main__":
    (X, Y) = (256, 256)
    make(step=3e-2, scale=0e+0, X=X, Y=Y)
    make(step=3e-2, scale=3e-1, X=X, Y=Y)
    make(step=3e-2, scale=5e-1, X=X, Y=Y)
    make(step=5e-2, scale=0e+0, X=X, Y=Y)
    make(step=5e-2, scale=3e-1, X=X, Y=Y)
    make(step=5e-2, scale=5e-1, X=X, Y=Y)
