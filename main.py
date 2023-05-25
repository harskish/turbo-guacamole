from pyviewer.toolbar_viewer import AutoUIViewer # pip install pyviewer
from pyviewer.params import *
import numpy as np
import torch
import matplotlib.pyplot
import matplotlib.cm
from multiprocessing import Lock

dev = 'cpu'
if torch.backends.mps.is_available():
    dev = 'mps'
if torch.cuda.is_available():
    dev = 'cuda'

valid_cmaps = []
for cmap in matplotlib.pyplot.colormaps():
    if hasattr(matplotlib.cm.get_cmap(cmap), 'colors'):
        valid_cmaps.append(cmap)

@strict_dataclass
class State(ParamContainer):
    res: Param = EnumSliderParam('Resolution scale', 0.75,
        [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0])
    cr: Param = FloatParam('C_real', -0.744, -2, 2) # -0.778
    ci: Param = FloatParam('C_imag',  0.148, -2, 2)
    max_iter: Param = IntParam('Iteration limit', 200, 1, 1_000)
    invert: Param = BoolParam('Inverted drawing', True)
    cmap: Param = EnumParam('Palette', 'twilight', valid_cmaps)
    cran: Param = Float2Param('Color range', (0.0, 1.0), 0.0, 1.0, overlap=False)

class Viewer(AutoUIViewer):
    def setup_state(self):
        self.state_lock = Lock()
        self.state = State()
        self.state_last = None
        self.curr_xform = np.eye(3)
        self.restart_rendering()

    def restart_rendering(self):
        W, H = ((self.output_pos_br - self.output_pos_tl) * self.state.res).astype(np.int32)
        S = np.diag([1.7, 1.7, 1]) # initial scale
        A = np.diag([1, H/W, 1]) # aspect ratio correction
        M = np.linalg.inv(self.pan_handler.get_transform_ndc())
        self.curr_xform = self.curr_xform @ M
        cnv_tl, cnv_br = (A @ S @ self.curr_xform @ np.array([(-1, -1, 1), (1, 1, 1)]).T).T[:2, :2]
        cnv_w, cnv_h = cnv_br - cnv_tl
        self.pan_handler.zoom = 1.0
        self.pan_handler.pan = (0, 0)

        # SoA-style data
        # Will be compacted as tasks finish
        self.curr_iter = 0
        self.colormap = torch.tensor(matplotlib.cm.get_cmap(self.state.cmap).colors, device=dev)
        self.posx = torch.arange(0, W, dtype=torch.int64, device=dev).tile(H)   # 0123-0123-0123...
        self.posy = torch.arange(0, H, dtype=torch.int64, device=dev).repeat_interleave(W) # 0000-1111-2222...
        self.zr = cnv_tl[0] + cnv_w * self.posx / (W - 1)
        self.zi = cnv_tl[1] + cnv_h * self.posy / (H - 1)
        self.image = torch.ones((H, W, 3), dtype=torch.float32, device=dev)
        self.image *= self.color_LUT(self.state.max_iter) if self.state.invert else self.color_LUT(0)

    def draw_toolbar(self):
        imgui.text(f'Active: {np.prod(self.zr.shape)} / {np.prod(self.image.shape)}')
        if imgui.button('Restart'):
            with self.state_lock:
                self.restart_rendering()

    # Lerped LUT
    def color_LUT(self, i):
        tmin, tmax = self.state.cran
        t = i / self.state.max_iter
        t = tmin + t * (tmax - tmin) # use part of range?
        t *= (len(self.colormap) - 1)
        lo = int(np.floor(t))
        hi = int(np.ceil(t))
        t_fract = t - lo
        return self.colormap[lo] * (1 - t_fract) + self.colormap[hi] * t_fract

    def compute(self):
        with self.state_lock:
            return self.process(self.state)

    def process(self, s: State):
        if self.curr_iter >= s.max_iter:
            return None # show previous image
        
        # Escape condition: |z| > 2
        valid = (self.zr**2 + self.zi**2) <= 4

        # Keep updating colors of non-escaped elements
        idx = ~valid if s.invert else valid
        self.image[self.posy[idx], self.posx[idx], :] = self.color_LUT(self.curr_iter)

        # Compact arrays
        self.zr = self.zr[valid]
        self.zi = self.zi[valid]
        self.posx = self.posx[valid]
        self.posy = self.posy[valid]

        # Update zs
        re = self.zr**2 - self.zi**2  # re(z^2)
        im = 2 * self.zr * self.zi    # im(z^2)
        self.zr = re + s.cr
        self.zi = im + s.ci

        # Update iteration counter
        self.curr_iter += 1

        # Return updated result
        return self.image

viewer = Viewer('Julia viz')