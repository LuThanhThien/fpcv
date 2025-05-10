from typing import Literal, List
import torch
from torch import nn
from torch.nn import functional as F
from binary_image.image import BinaryImage, config

class RasterScanner(nn.Module):
    def __init__(self):
        super(RasterScanner, self).__init__()
        self.reset()
        self._init_filters()
        
    def reset(self):
        self.eq_table = {0: 0}
        self.label_id = 1
        
    def _init_filters(self):
        # Accumulated filters
        acc_all     = torch.Tensor([[1, 1],
                                    [1, 0]],)
        acc_cornor  = torch.Tensor([[1, 0],
                                    [0, 0]],)
        acc_top     = torch.Tensor([[0, 1],
                                    [0, 0]],)
        acc_bot     = torch.Tensor([[0, 0],
                                    [1, 0]],)
        # Canceled filters
        can_diag    = torch.Tensor([[0, -1],
                                    [-1, 0]],)
        can_bot     = torch.Tensor([[0, 0],
                                    [-1, 0]],)
        
        self.filter = torch.stack([
            acc_all, acc_cornor, 
            acc_top, acc_bot,
            can_diag, can_bot
        ]).to(dtype=torch.int8)[:, None]
        
        
    def _run_conv(self, data, filter):
        """
        Run convolution with the given filter.
        """
        conv_result = nn.functional.conv2d(data.to(filter.dtype), filter, stride=1, padding=0)
        return conv_result

    def get_new_label(self):
        label = self.label_id
        self.label_id += 1
        return label
    
    def _get_min_label(self, label):
        if label not in self.eq_table:
            return label
        current_label = self.eq_table[label]
        if current_label == label:
            return current_label
        return self._get_min_label(current_label)

    def set_equivalent(self, label0, label1):
        label0 = int(label0)
        label1 = int(label1)
        root0 = self._get_min_label(label0)
        root1 = self._get_min_label(label1)
        min_label = min(root0, root1)
        self.eq_table[label0] = min_label
        self.eq_table[label1] = min_label
        return min_label

    def strategy(self, data, labels, window_x, window_y):
        # Re-assign to the tensor new part
        ## If upper of left border
        tx, ty = window_x + 1, window_y + 1
        if data[..., ty, tx] == 0:
            return

        label00 = labels[..., window_y, window_x]
        label10 = labels[..., window_y, window_x + 1]
        label01 = labels[..., window_y + 1, window_x]
        if label00 != 0:
            label = label00
            if label10 != 0 and label01 != 0:
                label = self.set_equivalent(label00, label10)
                label = self.set_equivalent(label, label01)
            elif label10 != 0:
                label = self.set_equivalent(label00, label10)
            elif label01 != 0:
                label = self.set_equivalent(label00, label01)         
        else:
            if label01 == label10 == 0:
                # print("New label")
                label = self.get_new_label()
            elif label10 != 0 and label01 != 0:
                label = self.set_equivalent(label10, label01)
            else:
                label = label10 if label10 else label01
        labels[..., ty, tx] = label
        # print(f"{tx}, {ty}: [{int(label00)} {int(label10)}] [{int(label01)}, {int(label)}]")
        # print(labels)

    def merge(self, output, window_x, window_y):
        current_label = int(output[..., window_y, window_x])
        output[..., window_y, window_x] = self._get_min_label(current_label)

    def iter_modify(self, data: BinaryImage, *args, **kwargs):
        self.reset()    
        # Pad to handle boundaries
        data = nn.functional.pad(data, [1, 1, 1, 1])
        height = data.height
        width = data.width
        labels = torch.zeros_like(data, dtype=torch.int32)
        
        # Raster Scanning
        for j in range(height - 1):
            for i in range(width - 1):
                # Apply the strategy
                self.strategy(data, labels, i, j)

        # Merge labels 
        for j in range(height - 1):
            for i in range(width - 1):
                # Merge 
                self.merge(labels, i, j)
                
        # Unpad
        labels = labels[..., 1:-1, 1:-1]

        return labels
    
    def get_id_mask(self, pos_mask: torch.Tensor):
        B, C, H, W = pos_mask.shape
        flat = pos_mask.view(B, C, -1)               # → (1,1,16)
        cum = torch.cumsum(flat, dim=-1)            # → (1,1,16), values go 1,1,1,…,2,…,3,…
        # zero‐out the positions that were originally 0
        flat_numbered = cum * flat                  # 1*1=1, 1*0=0, 2*1=2, etc.
        numbered_mask = flat_numbered.view(B, C, H, W) 
        return numbered_mask    

    def fast_modify(self, data: BinaryImage, intermediate: bool = False, *args, **kwargs):
        res_list = []
        res = nn.functional.pad(data, [1, 0, 1, 0]).to(torch.int8)
        last_res = res.clone()
        M = torch.ones_like(res, dtype=torch.int8)
        E = torch.zeros_like(res, dtype=torch.int8)
        
        while True:
            # Calculate the accumulated values
            res = res * M + E
            tmp = self._run_conv(res, self.filter)
            all_val = tmp[:, [0]]
            cornor_val = tmp[:, [1]]
            top_val = tmp[:, [2]]
            bot_val = tmp[:, [3]]
            can_diag = tmp[:, [4]]
            can_bot = tmp[:, [5]]
            
            # Id mask
            pos_mask = all_val == 0
            id_mask = self.get_id_mask(pos_mask * data)
            
            # Filter the values
            K = cornor_val > 0
            P = (top_val > 0) * (bot_val > 0)
            A = can_diag 
            B = can_bot 
            C = all_val + id_mask

            # New labels
            res = (A * K) + (B * P * ~K) + C 
            res *= data
            res = res * M[..., 1:, 1:] + E[..., 1:, 1:]

            # Repad and check for termination
            res = nn.functional.pad(res, [1, 0, 1, 0])
            if intermediate:
                res_list.append(res[..., 1:, 1:])
            
            # Check for convergence
            if torch.all(res == last_res):
                break

            # Calculate the next index
            last_res = res.clone()
            M = ~(P * ~K)
            E = bot_val * ~M
            M = M[..., 1:, 1:]
            E = E[..., 1:, 1:]
            M = nn.functional.pad(M, [0, 1, 1, 0], value=1)
            E = nn.functional.pad(E, [0, 1, 1, 0])
            M = nn.functional.pad(M, [1, 0, 1, 0], value=1)
            E = nn.functional.pad(E, [1, 0, 1, 0])
            
        # breakpoint()
        
        if intermediate:
            return res_list
        
        # Unpad
        # res = self._merge_fast(res)
        res = res[..., 1:, 1:]

        return res

    def forward(self,
                img: BinaryImage,
                mode: Literal['iterate', 'fast'] = 'iterate',
                *args, **kwargs):
        self.reset()
        # Pad to handle boundaries
        
        if mode == 'iterate':
            labels = self.iter_modify(img, *args, **kwargs)
        elif mode == 'fast':
            labels = self.fast_modify(img, *args, **kwargs)
        else:
            raise ValueError("Invalid mode. Use 'iterate' or 'fast'.")
        
        return labels
    
    def visualize(self, labels, save_path="labeled_output.png"):
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
        
        """
        Visualize labeled components by coloring each label differently and save the result.
        """
        labels_np = labels.squeeze().cpu().numpy()

        # Get unique non-zero labels
        unique_labels = sorted(np.unique(labels_np))
        unique_labels = [label for label in unique_labels if label != 0]

        # Create a mapping from label to color
        colormap = plt.cm.get_cmap('tab20', len(unique_labels))
        label_to_color = {0: (0, 0, 0)}  # Background is black
        for idx, label in enumerate(unique_labels):
            rgb = colormap(idx)[:3]  # Get RGB (0-1 range)
            rgb_255 = tuple(int(c * 255) for c in rgb)
            label_to_color[label] = rgb_255
        
        print(f"Unique labels: {unique_labels}")

        # Create color image
        height, width = labels_np.shape
        color_image = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                color_image[y, x] = label_to_color[labels_np[y, x]]

        # Save image
        image = Image.fromarray(color_image)
        image.save(save_path)
        print(f"Labeled image saved to {save_path}")
    
    def stream(self, frames: List[torch.Tensor], save_path="stream.mp4", fps: int = 2):
        """
        Create an MP4 showing each iteration of the fast labeling process.
        `frames` should be the list of intermediate label tensors returned when
        calling `fast_modify(..., intermediate=True)`.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import imageio

        # Convert tensors to numpy arrays
        frames_np = [f.squeeze().cpu().numpy() for f in frames]

        # Collect unique non-zero labels across all frames for consistent coloring
        unique_labels = sorted({int(l) for arr in frames_np for l in np.unique(arr) if l != 0})
        colormap = plt.cm.get_cmap('tab20', len(unique_labels))
        # Map each label to an RGB tuple
        label_to_color = {0: (0, 0, 0)}
        for idx, label in enumerate(unique_labels):
            rgb = colormap(idx)[:3]
            label_to_color[label] = tuple(int(c * 255) for c in rgb)

        # Initialize video writer
        writer = imageio.get_writer(str(save_path), fps=fps, format='mp4')
        for arr in frames_np:
            h, w = arr.shape
            img = np.zeros((h, w, 3), dtype=np.uint8)
            # Paint each pixel according to its label
            for lab, col in label_to_color.items():
                img[arr == lab] = col
            writer.append_data(img)
        writer.close()
        print(f"Stream saved to {save_path}")
