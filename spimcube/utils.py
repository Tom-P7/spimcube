import matplotlib.pyplot as plt
from matplotlib.widgets import Lasso
from matplotlib.path import Path
import numpy as np


class LassoSelectorImage:
    def __init__(self, image, ax_image):
        """Lasso selector to select points of an image.
        The points selected are displayed in another figure for check.
        Get the mask array of the selected points with ``mask_array`` attribute.

        Parameters
        ----------
        image : matplotlib.image.AxesImage
            An image return by matplotlib ``imshow``.
        ax_image : matplotlib axes
            The axes to which the image belong.

        """
        self.axes = ax_image
        self.canvas = ax_image.figure.canvas
        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)

        self.ny, self.nx = image.get_array().data.shape
        xx, yy = np.meshgrid(np.arange(0, self.nx), np.arange(0, self.ny))
        self.image_points = [(i, j) for (i, j) in zip(xx.ravel(), yy.ravel())]
        self._mask_array = None
        self.lasso = None

        # Display the selected area
        self.fig_, self.ax_ = plt.subplots()
        self.ax_.grid(b=False)
        self.ax_.set_ylim(0, self.ny)
        self.fig_.canvas.set_window_title("Selected area")

    def onpress(self, event):
        if self.canvas.widgetlock.locked():
            return
        if event.inaxes != self.axes:
            return
        if self.lasso:
            del self.lasso
        self.lasso = Lasso(self.axes, (event.xdata, event.ydata), self.callback)
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)

    def callback(self, verts):
        path = Path(verts)
        ind = path.contains_points(self.image_points)
        self.innercall(ind)
        self.canvas.widgetlock.release(self.lasso)

    def innercall(self, ind):
        """Do something with the points selected."""
        self._mask_array = ind.reshape(self.ny, self.nx)
        # Display the selected area
        self.ax_.imshow(self._mask_array)
        self.fig_.canvas.draw_idle()

    @property
    def mask_array(self):
        """Return the boolean mask array. True for selected point and False for unselected points."""
        return self._mask_array

    @mask_array.setter
    def mask_array(self, _):
        raise AttributeError("Value cannot be changed.")
