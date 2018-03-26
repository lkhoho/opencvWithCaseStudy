from skimage import feature


class HOG(object):
    """
    Histogram of Oriented Gradients (HOG).

    Computes a histogram over the orientation of the edges on a dense grid of uniformly-spaced cells.
    These cells can overlap and be contrast normalized to improve the accuracy of the descriptor.
    """

    def __init__(self, orientation=9, pixelsPerCell=(8, 8), cellsPerBlock=(3, 3), transform=False):
        """
        Construct a HOG instance.
        :param orientation: how many gradient orientations will be in each histogram (# of bins).
        :param pixelsPerCell: the number of pixels that will fall into each cell.
        :param cellsPerBlock: the number of cells that will fall into each block. Used for normalization of histograms.
        :param transform: optional power law compression (taking the log/square-root of the input image) before process.
        """

        self.orientation = orientation
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.transform = transform

    def describe(self, image):
        return feature.hog(image, orientations=self.orientation, pixels_per_cell=self.pixelsPerCell,
                           cells_per_block=self.cellsPerBlock, block_norm='L2', transform_sqrt=self.transform)
