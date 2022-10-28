import matplotlib.pyplot as plt
import numpy as np

class Visualize2D:
    def __init__(self, batched=True, x=4, y=1, ticks=False):
        self.x = x if batched else 1
        self.y = y if batched else 1

        self.cmap = 'plasma'

        self.fig = plt.figure()
        self.plots = []

        for i in range(x*y):
            ax = self.fig.add_subplot(y, x, i + 1, projection='rectilinear')

            if not ticks:
                ax.set_xticks([])
                ax.set_yticks([])

            self.plots.append(ax)

    def draw_images(self, images):
        images_squeezed = images[:, 0, :, :].detach().cpu().numpy()
        for i in range(images_squeezed.shape[0]):
            self.plots[i].imshow(images_squeezed[i], cmap = 'gray', interpolation = 'bicubic')

    def draw_points(self, point_lists):
        for count, points in enumerate(point_lists):
            cpu_points = points.detach().cpu().numpy()
            self.plots[count].scatter(cpu_points[:, 1], cpu_points[:, 0])

    def draw_heatmap(self, heat, heatmapaxis=0):
        heatmap_squeezed = heat[:, heatmapaxis, :, :].detach().cpu().numpy()
        for i in range(heatmap_squeezed.shape[0]):
            self.plots[i].imshow(heatmap_squeezed[i], cmap = self.cmap, interpolation = 'bicubic')

    def show(self):
        plt.show()
