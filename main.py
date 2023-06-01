from pyviewer.toolbar_viewer import AutoUIViewer # pip install pyviewer
from pyviewer.params import *
import numpy as np
import torch
import matplotlib.pyplot
import matplotlib.cm
from multiprocessing import Lock

# MPS works poorly atm
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

valid_cmaps = []
for cmap in matplotlib.pyplot.colormaps():
    if hasattr(matplotlib.cm.get_cmap(cmap), 'colors'):
        valid_cmaps.append(cmap)

# cr, ci, max_iter, [sx, sy, tx, ty], tmin, tmax, tpow, cmap
presets = [
    (-0.5370, -0.5260, 140, [1.000, 1.000, 0.000, 0.000], (0.0, 1.0), 1.0, "magma"),
    (-0.7540, -0.1030, 250, [1.000, 1.000, 0.000, 0.000], (0.0, 1.0), 1.3, "magma"),
    (-0.7780, 0.1480, 140, [1.000, 1.000, 0.000, 0.000], (0.0, 1.0), 1.0, "twilight_shifted_r"),
    (-0.7440, 0.1480, 140, [1.000, 1.000, 0.000, 0.000], (0.0, 1.0), 1.0, "twilight_shifted_r"),
    (-0.8460, -0.2170, 200, [0.121, 0.121, -0.049, 0.223], (0.0, 1.0), 1.0, "twilight_shifted_r"),
    (-0.5370, -0.5260, 140, [0.087, 0.087, -0.062, 0.096], (0.0, 1.0), 1.0, "twilight_shifted_r"),
    (-0.8890, -0.2220, 200, [1.001, 1.001, 0.007, -0.047], (0.0, 1.0), 0.8, "twilight_shifted_r"),
    (-0.7880, 0.1620, 140, [1.000, 1.000, 0.020, 0.044], (0.00, 1.00), 0.686, "twilight_shifted_r"),
    (-0.5000, -0.5260, 600, [1.000, 1.000, 0.000, 0.000], (0.00, 1.00), 0.569, "twilight_shifted_r"),
    (-0.5000, -0.5180, 600, [1.000, 1.000, 0.000, 0.000], (0.00, 1.00), 0.569, "twilight_shifted_r"),
    (-0.5370, -0.5000, 218, [1.000, 1.000, 0.000, 0.000], (0.27, 1.00), 0.232, "twilight_r"),
]

@strict_dataclass
class State(ParamContainer):
    res: Param = EnumSliderParam('Resolution scale', 1.0,
        [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0])
    cr: Param = FloatParam('C_real', -0.537, -1.1, 1.1)
    ci: Param = FloatParam('C_imag', -0.526, -1.1, 1.1)
    max_iter: Param = IntParam('Iteration limit', 140, 1, 1_000)
    invert: Param = BoolParam('Inverted drawing', True)
    cmap: Param = EnumParam('Palette', 'magma', sorted(valid_cmaps))
    cran: Param = Float2Param('Color range', (0.0, 1.0), 0.0, 1.0, overlap=False)
    tpow: Param = FloatParam('Palette exponent', 1.0, 0.1, 3.0)

class Viewer(AutoUIViewer):
    def setup_state(self):
        self.state_lock = Lock()
        self.state = State()
        self.state_last = None
        self.curr_xform = np.eye(3)
        self.cmap_cache = {}
        self.restart_rendering()

    def get_cmap_cached(self):
        if self.state.cmap not in self.cmap_cache:
            self.cmap_cache[self.state.cmap] = \
                torch.tensor(matplotlib.cm.get_cmap(self.state.cmap).colors, device=dev)
        return self.cmap_cache[self.state.cmap]

    def restart_rendering(self):
        self.state_lock.acquire()
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
        self.posx = torch.arange(0, W, dtype=torch.int64, device=dev).tile(H) # 0123-0123-0123...
        self.posy = torch.arange(0, H, dtype=torch.int64, device=dev).repeat_interleave(W) # 0000-1111-2222...
        self.zr = cnv_tl[0] + cnv_w * self.posx / (W - 1)
        self.zi = cnv_tl[1] + cnv_h * self.posy / (H - 1)
        self.counts = self.state.max_iter * torch.ones((H, W), dtype=torch.int32, device=dev)
        self.state_lock.release()

    def draw_toolbar(self):
        imgui.text(f'Alive: {100*np.prod(self.zr.shape)/np.prod(self.counts.shape[:2]):.2f}%')
        if imgui.button('Restart'):
            self.restart_rendering()

    def export_preset(self):
        s = self.state
        sx, sy, tx, ty = self.curr_xform.reshape(-1)[np.array([0, 4, 2, 5])]
        print(f'({s.cr:.4f}, {s.ci:.4f}, {s.max_iter}, [{sx:.3f}, {sy:.3f}, {tx:.3f}, {ty:.3f}], ({s.cran[0]:.2f}, {s.cran[1]:.2f}), {s.tpow:.3f}, "{s.cmap}"),')

    def load_preset(self, preset):
        with self.state_lock:
            s = self.state
            s.cr, s.ci, s.max_iter = preset[0:3]
            sx, sy, tx, ty = preset[3]
            self.curr_xform = np.array([sx, 0, tx, 0, sy, ty, 0, 0, 1]).reshape(3, 3)
            s.cran = preset[4]
            s.tpow = preset[5]
            s.cmap = preset[6]

    def draw_menu(self):
        with imgui.begin_menu('Presets', True) as file_menu:
            if file_menu.opened:
                if imgui.menu_item('Export', shortcut=None, selected=False, enabled=True)[0]:
                    self.export_preset()

                # submenu
                with imgui.begin_menu('Load', True) as open_recent_menu:
                    if open_recent_menu.opened:
                        for i, preset in enumerate(presets):
                            if imgui.menu_item(f'Preset {i+1}', None, False, True)[0]:
                                self.load_preset(preset)
                                self.restart_rendering()

    # Lerped LUT
    def color_LUT(self, i: torch.Tensor):
        cmap = self.get_cmap_cached()
        tmin, tmax = self.state.cran
        t = i / self.state.max_iter
        t = t ** self.state.tpow
        t = tmin + t * (tmax - tmin) # use part of range?
        t = t.clip(0, 1)
        t *= (len(cmap) - 1)
        lo = torch.floor(t).long()
        hi = torch.ceil(t).long()
        t_fract = (t - t.floor()).unsqueeze(-1)
        return cmap[lo] * (1 - t_fract) + cmap[hi] * t_fract

    def compute(self):
        with self.state_lock:
            return self.process(self.state)

    def process(self, s: State):
        if self.curr_iter >= s.max_iter:
            return self.color_LUT(self.counts)
        
        # Escape condition: |z| > 2
        valid = (self.zr**2 + self.zi**2) <= 4

        # Keep updating colors of non-escaped elements
        idx = ~valid if s.invert else valid
        self.counts[self.posy[idx], self.posx[idx]] = self.curr_iter

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
        return self.color_LUT(self.counts)

viewer = Viewer('Julia viz')