#!/usr/bin/python

"""
ungray.py
Camera to Window boundary discovery.
This app uses the first available camera as input and a window as output.
It calculates the boundaries between regions of differing flux separately
for the three RGB color planes.
"""

###############################################################################
__date__       = "20130131"
__author__     = "jlettvin"
__maintainer__ = "jlettvin"
__email__      = "jlettvin@gmail.com"
__copyright__  = "Copyright(c) 2013-2016 Jonathan D. Lettvin, All Rights Reserved"
__license__    = "GPL V3.0"
__status__     = "Production"
__version__    = "0.0.1"

#IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
# Imports from generic libraries
import          gc

from time       import (clock)
from scipy      import (zeros, array, asarray, arange, around)
from scipy      import (fabs, exp, tanh, sqrt, set_printoptions)
from optparse   import (OptionParser)
from itertools  import (product)

# Imports from OpenCV
try:
    from cv2        import (filter2D)
    from cv2.cv     import (GetSubRect, NamedWindow, GetCaptureProperty)
    from cv2.cv     import (CaptureFromCAM, CreateMat, GetMat)
    from cv2.cv     import (QueryFrame, ShowImage)
    from cv2.cv     import (WaitKey, DestroyAllWindows, CV_8UC3)
    from cv2.cv     import (CV_CAP_PROP_FRAME_WIDTH, CV_CAP_PROP_FRAME_HEIGHT)
except ImportError:
    print("Module OpenCV (cv2) is required")

#SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
# Change how scipy displays numbers.
set_printoptions(precision=2)

class Control(object):

    def __init__(self, name, limits, factor=1.1):
        assert isinstance(limits, tuple) or isinstance(limits, list)
        self.limit = list(limits)
        assert len(self.limit) == 3
        assert self.limit[0] <= self.limit[1] <= self.limit[2]
        assert self.limit[0] > 0.0
        self.factor = factor
        self.name = name
        self.todo = True

    def increment(self):
        self.todo = (self.limit[1] < self.limit[2])
        if isinstance(self.factor, int):
            self.limit[1] = min(self.limit[1]+self.factor, self.limit[2])
        else:
            self.limit[1] = min(self.limit[1]*self.factor, self.limit[2])
        self.show()

    def decrement(self):
        self.todo = (self.limit[0] < self.limit[1])
        if isinstance(self.factor, int):
            self.limit[1] = max(self.limit[1]-self.factor, self.limit[0])
        else:
            self.limit[1] = max(self.limit[1]/self.factor, self.limit[0])
        #self.limit[1] = max(self.limit*self.factor, self.limit[0])
        #self.limit[1] -= self.factor * self.todo
        #if self.limit[1] < self.limit[0]: self.limit[1] = self.limit[0]
        self.show()

    def show(self):
        print '%10s\t%2.3e\t<= %2.3e\t<=%2.3e' % (
                self.name, self.limit[0], self.limit[1], self.limit[2])

    @property
    def val(self):
        return self.limit[1]


class UnGray(object):

    #11111111111111111111111111111111111111111111111111111111111111111111111111
    def __init__(self, **kw):

        #2222222222222222222222222222222222222222222222222222222222222222222222
        def initOpenCV():
            # Setup OpenCV
            self.camera     = self.kw.get("camera", 0)
            self.capture    = CaptureFromCAM(self.camera)

            self.dtype      = float
            # A hack to prevent saturation.
            self.dsize      = tuple(int(GetCaptureProperty(self.capture, p))
                    for p in (CV_CAP_PROP_FRAME_WIDTH, CV_CAP_PROP_FRAME_HEIGHT))
            self.X, self.Y  = self.dsize
            self.shape      = (self.Y, self.X, 3)
            self.paste      = CreateMat(self.Y, self.X, CV_8UC3)
            self.target     = asarray(GetSubRect(self.paste, (0,0, self.X,self.Y)))

            # Associate color planes with correct letter.
            self.plane      = {c:n for n,c in enumerate('BGR')}

        self.control        = {
                ord('d'): Control('dxp'   , (1e-1, 1e-1, 2e+0), 1.1),
                ord('e'): Control('exp'   , (1e-2, 5e+0, 1e+1), 1.1),
                ord('r'): Control('radius', (   2,    6,   10), 1  ),
                ord('s'): Control('slope' , (1e+0, 5e+1, 1e+2), 1.1),
                'key':  ord('s'),
                }

        for k,v in self.control.iteritems():
            if k is 'key':
                continue
            v.show()

        self.newparams      = False # Set this true to read the JSON file.
        self.test           = False #True
        self.kw             = kw
        self.iteration      = 0
        self.scale          = 255.0
        self.char           = ' '
        self.elast          = 0
        rcontrol = self.control[ord('r')]
        self.edge           = 2*rcontrol.limit[2]+1
        self.rlast          = 0
        self.active         = kw.get('active', True)
        self.dynamic        = kw.get('dynamic', True)
        self.testR, self.testG, self.testB, self.testW = [], [], [], []
        initOpenCV()

    #11111111111111111111111111111111111111111111111111111111111111111111111111
    def rekernel(self):
        dcontrol = self.control[ord('d')]
        econtrol = self.control[ord('e')]
        rcontrol = self.control[ord('r')]
        radius = rcontrol.val
        dvalue = dcontrol.val
        evalue = econtrol.val
        rmax   = rcontrol.limit[2]
        if self.rlast != radius:
            inner, outer = float(radius-1), float(radius)
            shape      = (self.edge, self.edge)
            self.radii = list(product(arange(-rmax,rmax+1,1.0), repeat=2))
            self.radii = array([sqrt(x*x+y*y) for x,y in self.radii]).reshape(shape)

            if True:
                self.negative = -exp(-dvalue*(self.radii-outer)**2)
                self.positive = +exp(-dvalue*(self.radii-inner)**2)
            else:
                self.radii = around(self.radii)
                self.negative = zeros((self.edge,self.edge),dtype=float)
                self.negative[self.radii == outer] = -1.0
                self.positive = zeros(shape,dtype=float)
                self.positive[self.radii == inner] = +1.0

            self.negative /= fabs(self.negative.sum())
            self.positive /= fabs(self.positive.sum())

            self.kernel = self.negative + self.positive
            self.rlast = radius
        if self.elast != evalue:
            self.gauss = exp(-evalue * self.radii**2)
            self.gauss /= self.gauss.sum()
            self.elast = evalue

    #11111111111111111111111111111111111111111111111111111111111111111111111111
    def __call__(self, **kw):
        # Fetch the camera image.
        self.rekernel()
        self.iteration += 1

        if not self.test:
            self.source     = asarray(
                    GetMat(QueryFrame(self.capture))).astype(float)
        else:
            # clear to black
            self.source[:,:,:] = zeros((self.Y,self.X,3))

            # change every two seconds.
            value = 255.0 if (int(clock())/2) % 2 == 1 else 0.0

            if self.kw['image']:
                pass
            else:
                # Put up blinking white dots
                #self.source[y2   ,x2   ,:] = value
                for x,y in self.testW: self.source[y,x,:] = value
                for x,y in self.testR: self.source[y,x,self.plane['R']] = value
                for x,y in self.testG: self.source[y,x,self.plane['G']] = value
                for x,y in self.testB: self.source[y,x,self.plane['B']] = value

        # Normalize input
        self.source /= self.scale
        # Run the chosen function.
        if self.dynamic:
            #self.target = (1.0+tanh(4.0 * self.source)) * self.scale / 2.0
            #self.target = self.source / 2.0
            self.target[:,:,:] = self.source[:,:,:]
        else:
            (self.full if self.active else self.noop)()
        # Display the result.
        ShowImage("ungray", self.paste)
        return self.listen()

    #11111111111111111111111111111111111111111111111111111111111111111111111111
    def report(self, msg):
        print ' '*79 + '\r' + msg

    #11111111111111111111111111111111111111111111111111111111111111111111111111
    def listen(self):
        result = True
        # Check for keyboard input.
        key = WaitKey(6)    # milliseconds between polling (-1 if none).
        if key == -1:
            gc.collect()
            pass
        elif key == 27:
            result = False
        elif ord('-') == key or ord('_') == key:
            self.control[self.control['key']].decrement()
        elif ord('+') == key or ord('=') == key:
            self.control[self.control['key']].increment()
        elif key in self.control.keys():
            self.control['key'] = key
        elif ord(' ') == key:
            self.active ^= True
            self.report('processing' if self.active else 'noop')
        elif ord('.') == key:
            self.test ^= True
            self.report('test pattern: ' + str(self.test))
        elif ord('!') == key and not self.newparams:
            self.saturate ^= True
            self.report('saturate: ' + str(self.saturate))
        elif ord('a') <= key <= ord('z'):
            pass
        elif ord('1') <= key <= ord('8') and not self.newparams:
            self.pupil = key - ord('0')
            self.aperture = self.pupil * 1e-3
            self.report(
                    'Pupil: %d millimeters = %f' %
                    (self.pupil, self.aperture))
        else:
            #self.report('Unknown: %d' % (key))
            pass
        return result

    #11111111111111111111111111111111111111111111111111111111111111111111111111
    def full(self):
        self.ungray()
        return self

    #11111111111111111111111111111111111111111111111111111111111111111111111111
    def ungray(self):
        self.R = self.source[:,:,0]
        self.G = self.source[:,:,1]
        self.B = self.source[:,:,2]

        self.R = filter2D(self.R, -1, self.kernel)
        self.G = filter2D(self.G, -1, self.kernel)
        self.B = filter2D(self.B, -1, self.kernel)

        slope = self.control[ord('s')].val
        self.R = (1.0 + tanh(slope*self.R)) / 2.0
        self.G = (1.0 + tanh(slope*self.G)) / 2.0
        self.B = (1.0 + tanh(slope*self.B)) / 2.0

        self.R = filter2D(self.R, -1, self.gauss)
        self.G = filter2D(self.G, -1, self.gauss)
        self.B = filter2D(self.B, -1, self.gauss)

        self.target[:,:,0] = self.R * self.scale
        self.target[:,:,1] = self.G * self.scale
        self.target[:,:,2] = self.B * self.scale

    #11111111111111111111111111111111111111111111111111111111111111111111111111
    def noop(self):
        self.target[:,:,:] = self.source[:,:,:] * self.scale

#MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option( "-c", "--camera",
            type=int, default=0,
            help="choose a camera by int (default 0)")
    parser.add_option( "-d", "--dynamic",
            action="store_true", default=False,
            help="test dynamic range experiment")
    parser.add_option( "-i", "--image",
            type=str, default=None,
            help="use an image in place of the test pattern")
    parser.add_option( "-v", "--verbose",
            action="store_true", default=False,
            help="announce actions and sizes")
    (opts, args) = parser.parse_args()
    kw = vars(opts)

    try:
        NamedWindow("Human refraction and diffraction", 1)
        ungray = UnGray(**kw)
        while ungray(): pass
    finally:
        DestroyAllWindows()
