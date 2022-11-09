import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

    # We assume that argmax was used on the segmentation before.
    def draw_segmentation(self, segmentation, num_classes, opacity=1.0):
        segmentations_squeezed = segmentation.squeeze().detach().cpu().numpy()
        colors = [np.array(cm.get_cmap(self.cmap)(i*(1/num_classes))[0:3]) for i in range(num_classes)]

        colored = self.class_to_color(segmentations_squeezed, colors)

        for i in range(colored.shape[0]):
            self.plots[i].imshow(np.moveaxis(colored[i], 0, -1), cmap = 'gray', interpolation = 'bicubic', alpha=opacity)



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

    def class_to_color(self, prediction, class_colors):
        prediction = np.expand_dims(prediction, 1)
        output = np.zeros((prediction.shape[0], 3, prediction.shape[-2], prediction.shape[-1]), dtype=np.float)
        for class_idx, color in enumerate(class_colors):
            mask = class_idx == prediction
            curr_color = color.reshape(1, 3, 1, 1)
            segment = mask*curr_color # should have shape 1, 3, 100, 100
            output += segment

        return output
    
    def show(self):
        plt.show()