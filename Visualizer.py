import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
plt.ioff()


class Visualize2D:
    def __init__(self, batched=True, x=4, y=1, ticks=False, remove_border = False, do_not_open = False):
        if do_not_open:
            plt.ioff()
        else:
            plt.ion()


        self.x = x if batched else 1
        self.y = y if batched else 1

        self.cmap = 'plasma'

        self.fig = plt.figure()
        self.plots = []

        
        if x==1 and y==1 and remove_border:
            ax = plt.Axes(self.fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            self.fig.add_axes(ax)
            self.plots.append(ax)
            plt.axis('off')
            return

        for i in range(x*y):
            ax = self.fig.add_subplot(y, x, i + 1, projection='rectilinear')

            if not ticks:
                ax.set_xticks([])
                ax.set_yticks([])

            self.plots.append(ax)

    # We assume that argmax was used on the segmentation before.
    def draw_segmentation(self, segmentation, num_classes, opacity=1.0):
        segmentation = segmentation.detach().cpu().numpy()
        colors = [np.array(cm.get_cmap(self.cmap)(i*(1/num_classes))[0:3]) for i in range(num_classes)]

        colored = self.class_to_color(segmentation, colors)

        for i in range(colored.shape[0]):
            self.plots[i].imshow(np.moveaxis(colored[i], 0, -1), cmap = 'gray', interpolation = 'bicubic', alpha=opacity)

    def draw_segmenation_sequence(self, segmentations, num_classes, opacity=1.0):
        segmentations = segmentations.detach().cpu().numpy()
        for s in range(segmentations.shape[0]):
            segmentation = segmentations[0]
            colors = [np.array(cm.get_cmap(self.cmap)(i*(1/num_classes))[0:3]) for i in range(num_classes)]

            colored = self.class_to_color(segmentation[None, :, :], colors)

            for i in range(colored.shape[0]):
                self.plots[s].imshow(np.moveaxis(colored[i], 0, -1), cmap = 'gray', interpolation = 'bicubic', alpha=opacity)


    def draw_sequence(self, images, axis=0):
        images_squeezed = images[axis, :, :, :].detach().cpu().numpy()
        for i in range(images_squeezed.shape[0]):
            self.plots[i].imshow(images_squeezed[i], cmap = 'gray', interpolation = 'bicubic')


    def draw_images(self, images):
        images_squeezed = images[:, 0, :, :].detach().cpu().numpy()
        for i in range(images_squeezed.shape[0]):
            self.plots[i].imshow(images_squeezed[i], cmap = 'gray', interpolation = 'bicubic')

    def draw_points(self, point_lists, color='blue'):
        for count, points in enumerate(point_lists):
            cpu_points = points
            self.plots[count].scatter(cpu_points[:, 1], cpu_points[:, 0], color=color)

    def draw_heatmap(self, heat, heatmapaxis=0, opacity=1.0):
        heatmap_squeezed = heat[:, heatmapaxis, :, :].detach().cpu().numpy()
        for i in range(heatmap_squeezed.shape[0]):
            self.plots[i].imshow(heatmap_squeezed[i], cmap = self.cmap, interpolation = 'bicubic', alpha=opacity)

    def class_to_color(self, prediction, class_colors):
        prediction = np.expand_dims(prediction, 1)
        output = np.zeros((prediction.shape[0], 3, prediction.shape[-2], prediction.shape[-1]), dtype=np.float)
        for class_idx, color in enumerate(class_colors):
            mask = class_idx == prediction
            curr_color = color.reshape(1, 3, 1, 1)
            segment = mask*curr_color # should have shape 1, 3, 100, 100
            output += segment

        return output
    
    def get_as_numpy_arr(self):
        for ax in self.plots:
            ax.axis('tight')
            
        plt.subplots_adjust(0,0,1,1,0,0)
        width, height = self.fig.get_size_inches() * self.fig.get_dpi()
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    def show(self):
        self.fig.show()

    def close(self):
        plt.close()