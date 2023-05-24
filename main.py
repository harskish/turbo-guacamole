from pyviewer.toolbar_viewer import AutoUIViewer
from pyviewer.params import *
import numpy as np
import torch
import matplotlib.cm
from multiprocessing import Lock

dev = 'cuda'

@strict_dataclass
class State(ParamContainer):
    W: Param = EnumSliderParam('W', 8, [8, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    H: Param = EnumSliderParam('H', 8, [8, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    cr: Param = FloatParam('C_real', -0.744, -2, 2)
    ci: Param = FloatParam('C_imag',  0.148, -2, 2)
    max_iter: Param = IntParam('Iteration limit', 200, 1, 1_000)
    invert: Param = BoolParam('Inverted drawing', True)

class Viewer(AutoUIViewer):
    def setup_state(self):
        self.state_lock = Lock()
        self.state = State()
        self.state_last = None
        self.colormap = torch.tensor(matplotlib.cm.get_cmap("twilight").colors, device=dev)
        self.restart_rendering()

    def restart_rendering(self):
        # SoA-style data
        # Will be compacted as tasks finish
        self.curr_iter = 0
        self.posx = torch.arange(0, self.state.W, dtype=torch.int64, device=dev).tile(self.state.H)   # 0123-0123-0123...
        self.posy = torch.arange(0, self.state.H, dtype=torch.int64, device=dev).repeat_interleave(self.state.W) # 0000-1111-2222...
        self.z = self.posx / (self.state.W - 1) + 1j*self.posy / (self.state.H - 1) # in [ 0, 1]^2
        self.z = 4*self.z - 2*(1 + 1j) # in [-2, 2]^2
        self.image = torch.ones((self.state.H, self.state.W, 3), dtype=torch.float32, device=dev)
        self.image *= self.color_LUT(self.state.max_iter) if self.state.invert else self.color_LUT(0)

    def draw_toolbar(self):
        imgui.text(f'Active: {np.prod(self.z.shape)} / {np.prod(self.image.shape)}')
        if imgui.button('Restart'):
            with self.state_lock:
                self.restart_rendering()

    # Lerped LUT
    def color_LUT(self, i):
        t = i / self.state.max_iter
        t *= 0.6 # use part of range?
        t *= (len(self.colormap) - 1)
        lo = int(np.floor(t))
        hi = int(np.ceil(t))
        t_fract = t - lo
        return self.colormap[lo] * (1 - t_fract) + self.colormap[hi] * t_fract

    def compute(self):
        with self.state_lock:
            params = { k: p.value for k, p in self.state }
            return self.process(**params)

    def process(
        self,
        W: int,
        H: int,
        cr: float,
        ci: float,
        max_iter: int,
        invert: bool,
    ):
        if self.curr_iter >= max_iter:
            return None # show previous image
        
        # Escape condition: |z| > 2
        valid = torch.absolute(self.z) <= 2

        # Keep updating colors of non-escaped elements
        idx = ~valid if invert else valid
        self.image[self.posy[idx], self.posx[idx], :] = self.color_LUT(self.curr_iter)

        # Compact arrays
        self.z = self.z[valid]
        self.posx = self.posx[valid]
        self.posy = self.posy[valid]

        # Update zs
        self.z = self.z**2 + (cr+ci*1j)

        # Update iteration counter
        self.curr_iter += 1

        # Return updated result
        return self.image

viewer = Viewer('Julia viz')