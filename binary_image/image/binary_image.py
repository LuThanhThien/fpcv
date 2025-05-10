from functools import cached_property
import warnings 
from typing import Union, Optional
from pathlib import Path
from PIL import Image, ImageDraw
import subprocess
import math
import numpy as np
import torch
from torch.nn import functional as F

__all__ = [
    "load_binary_image",
    "BinaryImage",
]

warnings.filterwarnings("ignore", category=UserWarning)

def load_binary_image(path: Path, device="cpu") -> torch.Tensor:
    assert path.exists(), f"{path} does not exist."

    # Pillow understands GIF natively
    img = Image.open(path).convert("L")          # 8‑bit grayscale
    arr = np.array(img)                          # H × W, uint8

    arr = (arr > 127).astype(np.float32)         # strict binary 0/1
    tensor = torch.from_numpy(arr).unsqueeze(0)  # 1 × H × W

    return tensor.to(device)

class BinaryImage(torch.Tensor):
    
    @staticmethod
    def load(path, *args, **kwargs) -> "BinaryImage":
        path = Path(str(path)).resolve()
        data = load_binary_image(path)
        return BinaryImage(data, *args, **kwargs)
    
    def detect_components(self):
        """ Detect connected components in the image """
        pass

    def num_components(self):
        """ Return number of connected components in the image """
        pass
    
    @property
    def height(self):
        return self.data.shape[-2]
    
    @property
    def width(self):
        return self.data.shape[-1]
        
    @cached_property
    def area(self):
        """ Return total area of connected components """
        return torch.sum(self.data).item()
    
    @cached_property
    def meshgrid(self):
        """ Return meshgrid of the image """
        X = torch.arange(self.width, device=self.data.device, dtype=torch.float32)
        Y = torch.arange(self.height, device=self.data.device, dtype=torch.float32)
        Y, X = torch.meshgrid(Y, X)
        return X.flatten(), Y.flatten()
    
    @cached_property
    def centers(self):
        """ Return list of centers of each connected components """
        Y, X = self.meshgrid
        data = self.data.flatten()
        if self.area == 0:
            return torch.tensor([0, 0], dtype=torch.int32)
        x = torch.sum(X * data) / self.area
        y = torch.sum(Y * data) / self.area
        out = torch.stack([x, y], dim=0).to(torch.int32)
        return out
    
    @cached_property
    def inertia(self):
        center = self.centers
        center_x = center[0]
        center_y = center[1]
        Y, X = self.meshgrid
        X, Y = X - center_x, Y - center_y
        data = self.data.flatten()
        Ix = torch.sum(X**2*data)
        Iy = torch.sum(Y**2*data)
        Ixy = torch.sum(2*X*Y*data)
        out = torch.stack([torch.stack([Ix, Ixy]), 
                           torch.stack([Ixy, Iy])], dim=1)
        return out

    def _second_derivative(self, a, b, c, theta):
            two_theta = 2 * theta
            return (a - c) * torch.sin(two_theta) - b * torch.cos(two_theta)
    
    def _second_moment(self, theta):
        center = self.centers
        center_x = center[0]
        center_y = center[1]
        Y, X = self.meshgrid
        X, Y = X - center_x, Y - center_y
        data = self.data.flatten()
        return torch.sum(
            (X*torch.sin(theta) - Y*torch.cos(theta))**2 * data 
        )
        
    @cached_property 
    def second_moment_angles(self):
        inertia = self.inertia
        inertia_x = inertia[0, 0]
        inertia_y = inertia[1, 1]
        inertia_xy = inertia[0, 1]
        tan = inertia_xy / (inertia_x - inertia_y)
        theta1 = torch.arctan(tan) / 2
        theta2 = theta1 + torch.pi/2
        return theta1, theta2
    
    @cached_property
    def minimum_second_moment_angle(self):
        inertia = self.inertia
        inertia_x = inertia[0, 0]
        inertia_y = inertia[1, 1]
        inertia_xy = inertia[0, 1]
        theta1, theta2 = self.second_moment_angles
        if self._second_derivative(inertia_x, inertia_y, inertia_xy, theta1) > 0:
            return theta1
        else:
            return theta2

    @cached_property
    def maximum_second_moment_angle(self):
        inertia = self.inertia
        inertia_x = inertia[0, 0]
        inertia_y = inertia[1, 1]
        inertia_xy = inertia[0, 1]
        theta1, theta2 = self.second_moment_angles
        if self._second_derivative(inertia_x, inertia_y, inertia_xy, theta1) < 0:
            return theta1
        else:
            return theta2

    @cached_property
    def orientation_angle(self):
        return self.minimum_second_moment_angle
    
    @cached_property
    def roundness(self):
        """ Return list of roundness of each connected components """
        theta1 = self.orientation_angle
        theta2 = theta1 + torch.pi / 2
        E1 = self._second_moment(theta1)
        E2 = self._second_moment(theta2)
        return E1/E2
    
    @cached_property
    def boundaries(self):
        """ Return list of boundaries of each connected components """
        pass
    
    @cached_property
    def component_areas(self):
        """ Return list of areas of each connected components """
        pass

    @cached_property
    def euler_number(self):
        """ Return Euler number of the image """
        fuse_filter = torch.Tensor([
            [[1, 1],
             [1, 1]], 
            [[1, 0],
             [0, 0]],
            [[1, 1],
             [1, 0]]
        ])[:, None].to(torch.int8)
        vals = torch.Tensor([[3], [1]])[:, None]
        padded_data = F.pad(self.data, (1, 0, 1, 0), mode="constant", value=0)
        conv = F.conv2d(padded_data[None].to(torch.int8), fuse_filter, stride=1)
        conv = (conv[:, [0]] == vals) * (conv[:, 1:] == 0)
        conv = conv.sum(dim=-1).sum(dim=-1).flatten()
        eul = conv[1] - conv[0]
        return eul.item()
    
    def visualize(
        self,
        *,                                       # ── keyword‑only parameters
        save_path: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
        show_window: bool = False,
        draw_centroid: bool = True,
        draw_orientation: bool = True,
        draw_boundaries: bool = True,            # placeholder for future feature
        segmentation: bool = True,               # placeholder for future feature
        write_info_box: bool = False,
        arrow_len: int = 20,
        fg_colour: str = "blue",
        centroid_colour: str = "red",
    ) -> Image.Image:
        """
        Render the binary image with optional overlays.

        Parameters
        ----------
        save_path : str | pathlib.Path | None
            Where to save the rendered image.  If ``None`` the file is not saved.
        overwrite : bool, default False
            If ``False`` and *save_path* exists, raise an error instead of clobbering.
        show_window : bool, default False
            Call Pillow's ``Image.show()`` (and fall back to *wslview* if available).
        draw_centroid, draw_orientation, draw_boundaries, segmentation :
            Toggle each overlay layer.
        write_info_box : bool, default False
            Adds a small white rectangle with roundness and area.
        arrow_len : int, default 20
            Length in pixels of the orientation arrow half‑span.
        fg_colour, centroid_colour : str
            Colours used for drawing primitives.

        Returns
        -------
        PIL.Image.Image
            The rendered RGB image (useful for inline Jupyter display).
        """

        # ------------------------------------------------------------------
        # Prepare a base RGB canvas exactly once
        # ------------------------------------------------------------------
        img_np: np.ndarray = (self.data.squeeze().cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, mode="L").convert("RGB")
        draw = ImageDraw.Draw(img_pil)

        # ------------------------------------------------------------------
        # Pre‑compute common quantities (one property lookup each)
        # ------------------------------------------------------------------
        cx, cy = self.centers.tolist()           # floats, centroid in (x, y)
        area_val = float(self.area)

        # ------------------------------------------------------------------
        # 1) centroid
        # ------------------------------------------------------------------
        if draw_centroid:
            r = 2
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=centroid_colour)
            draw.text((cx + r + 1, cy + r + 1), f"({int(cx)}, {int(cy)})",
                    fill=centroid_colour)

        # ------------------------------------------------------------------
        # 2) principal‑axis orientation arrow
        # ------------------------------------------------------------------
        if draw_orientation:
            theta = float(self.orientation_angle)           # radians → float
            dx = arrow_len * math.cos(theta)
            dy = arrow_len * math.sin(theta)
            draw.line((cx, cy, cx + dx, cy + dy), fill=fg_colour, width=2)
            draw.line((cx, cy, cx - dx, cy - dy), fill=fg_colour, width=2)

        # ------------------------------------------------------------------
        # 3) extra info box
        # ------------------------------------------------------------------
        if write_info_box:
            info_lines = [
                f"Area: {area_val:.0f}",
                f"Roundness: {getattr(self, 'roundness', 'N/A')}",
            ]
            line_h = 12
            pad = 4

            box_w = max(len(s) for s in info_lines) * 6 + 2 * pad
            box_h = len(info_lines) * line_h + 2 * pad
            x0, y0 = self.width + pad, pad
            x1, y1 = x0 + box_w, y0 + box_h
            
            # create a wider canvas
            extra = box_w + 2 * pad
            new_w = self.width + extra
            canvas = Image.new("RGB", (new_w, self.height), "black")
            canvas.paste(img_pil, (0, 0))      # original image at left
            img_pil = canvas
            draw = ImageDraw.Draw(img_pil)
            
            draw.rectangle((x0, y0, x1, y1), fill="white")
            for k, txt in enumerate(info_lines):
                draw.text((x0 + pad, y0 + pad + k * line_h), txt, fill="black")

        # ------------------------------------------------------------------
        # Saving
        # ------------------------------------------------------------------
        if save_path is not None:
            save_path = Path(save_path)
            if save_path.exists() and not overwrite:
                raise FileExistsError(f"{save_path} already exists; set overwrite=True.")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img_pil.save(save_path)
            print(f"Saved to {save_path}")

        # ------------------------------------------------------------------
        # Display in a window if requested
        # ------------------------------------------------------------------
        if show_window:
            if not img_pil.show():
                # Pillow failed to open a GUI viewer -> try wslview (WSL) as fallback
                try:
                    subprocess.Popen(['wslview', str(save_path or '/tmp/preview.png')])
                except FileNotFoundError:
                    print("No GUI viewer available to display the image.")

        return img_pil
