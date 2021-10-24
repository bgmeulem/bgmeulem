import matplotlib.colors as clrs
import matplotlib.cm as cm
import matplotlib.collections as mcoll
from black_hole_math import *

plt.style.use('fivethirtyeight')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # six fivethirtyeight themed colors


class BlackHole:
    def __init__(self, mass=1, inclination=10, acc=10e-8):
        """Initialise black hole with mass and accretion rate
        Set viewer inclination above equatorial plane
        """
        self.t = inclination * np.pi / 180
        self.M = mass
        self.acc = acc  # accretion rate
        self.isoradial_properties = {'start_angle': 0,
                                     'angular_precision': 60}
        self.solver_params = {'root_precision': 25,
                              'iterations': 4,
                              'plot_inbetween': False,
                              'minP': 3}
        self.plot_params = {'save_plot': True,
                            'plot_ellipse': False,
                            'key': "",
                            'face_color': 'black',
                            'line_color': 'white',
                            'text_color': 'white',
                            'alpha': 1.,
                            'show_grid': False,
                            'legend': False,
                            'title': "Isoradials for M = {}".format(self.M)}
        self.isoradials = {}

    def setInclination(self, incl):
        self.t = incl * np.pi / 180

    def getIsoradial(self, radius, order):
        """Returns observer coordinates of isoradioal photon beams emitted at radius r of the black hole"""
        angles = []
        radii = []
        start_angle = self.isoradial_properties['start_angle']
        angular_precision = self.isoradial_properties['angular_precision']
        for alpha_ in tqdm(np.linspace(start_angle, np.pi, angular_precision),
                           desc='R = {}'.format(radius), position=2, leave=False):
            P_ = findP(r=radius, incl=self.t, alpha=alpha_, M=self.M, n=order, **self.solver_params)
            if len(P_):
                angles.append(alpha_)
                b_ = [float(b(P_, self.M)) for P_ in P_]
                radii.append(b_)
        if order > 0:
            angles = [a_ + np.pi for a_ in angles]

        # add second half of image
        angles += [2 * np.pi - a_ for a_ in angles[::-1]]
        radii += radii[::-1]

        # flip image if necessary
        if self.t > np.pi/2:
            angles = [a_ + np.pi for a_ in angles]

        self.isoradials[radius] = [angles, radii]
        return angles, radii

    def getRedshifts(self, isoradial, radius):
        angles, radii = isoradial
        redshifts = [redshift_factor(radius, angle, self.t, self.M, b_[0]) for b_, angle in zip(radii, angles)]
        return redshifts

    def plotIsoradial(self, direct_r: [], ghost_r: []):
        """Given an array of radii for the direct image and/or ghost image, plots the corresponding
        isoradials.
        Calculates the isoradials according to self.root_params
        Plots the isoradials according to self.plot_params"""

        def plotSingleIsoradial(isoradial_, ax_, plot_params, redshifts=[], colornorm=(0, 1)):
            def make_segments(x, y):
                """
                Create list of line segments from x and y coordinates, in the correct format
                for LineCollection: an array of the form numlines x (points per line) x 2 (x
                and y) array
                """

                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                return segments

            def colorline(
                    _ax, x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(*colornorm),
                    linewidth=3):
                """
                http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
                http://matplotlib.org/examples/pylab_examples/multicolored_line.html
                Plot a colored line with coordinates x and y
                Optionally specify colors in the array z
                Optionally specify a colormap, a norm function and a line width
                """

                # Default colors equally spaced on [0,1]:
                if z is None:
                    z = np.linspace(0.0, 1.0, len(x))

                # Special case if a single number:
                if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
                    z = np.array([z])

                z = np.asarray(z)

                segments = make_segments(x, y)
                lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm,
                                          linewidth=linewidth, alpha=self.plot_params['alpha'])
                lc.set_array(z)
                _ax.add_collection(lc)
                mx = max(segments[:][:, 1].flatten())
                _ax.set_ylim((0, mx))
                return _ax

            angles, radii = isoradial_

            # Plot isoradial
            for i, r_ in enumerate(np.array(radii).T):
                label = plot_params['key'] if i == 0 else None
                if len(redshifts):
                    ax_ = colorline(ax_, angles, r_, z=redshifts, cmap=cm.get_cmap('RdBu'))
                else:
                    ax_.plot(angles, r_, label=label, color=plot_params['line_color'],
                             alpha=plot_params['alpha'])
            if self.plot_params['legend']:
                plt.legend(prop={'size': 16})
            return ax_

        def plotEllipse(r_, ax):
            ax_ = ax
            a = np.linspace(-np.pi, np.pi, 2 * self.plot_params['angular_precision'])
            scale = 1. / mpmath.acos(cos_gamma(0, self.t))
            ell = [ellipse(r_ * scale, a_, self.t) for a_ in a]
            ax_.plot(a, ell)
            return ax_

        fig = plt.figure(figsize=(10, 10))
        axes = fig.add_subplot(111, projection='polar')
        axes.set_theta_zero_location("S")
        fig.patch.set_facecolor(self.plot_params['face_color'])
        axes.set_facecolor(self.plot_params['face_color'])
        if self.plot_params['show_grid']:
            axes.grid(color='grey')
            axes.tick_params(which='both', labelcolor=self.plot_params['text_color'],
                             labelsize=15)
        else:
            axes.grid()
            axes.spines['polar'].set_visible(False)

        # plot ghost images
        self.plot_params['line_color'] = 'grey'
        self.plot_params['alpha'] = .5
        for radius in tqdm(sorted(ghost_r), desc='Ghost Image', position=1, leave=False):
            self.plot_params['key'] = 'R = {}'.format(radius)
            isoradial = self.getIsoradial(radius, 1)
            redshifts = self.getRedshifts(isoradial, radius)
            axes = plotSingleIsoradial(isoradial, axes, self.plot_params, redshifts=redshifts)

        # plot direct images
        self.plot_params['line_color'] = 'white'
        self.plot_params['alpha'] = 1.
        for i, radius in enumerate(tqdm(sorted(direct_r), desc='Direct Image', position=1, leave=False)):
            self.plot_params['key'] = 'R = {}'.format(radius)
            isoradial = self.getIsoradial(radius, 0)
            redshifts = self.getRedshifts(isoradial, radius)
            if i == 0:
                colornorm = (-max(redshifts), max(redshifts))
            axes = plotSingleIsoradial(isoradial, axes, self.plot_params, redshifts=redshifts, colornorm=colornorm)

        if self.plot_params['plot_ellipse']:  # plot ellipse
            for radius in direct_r:
                axes = plotEllipse(radius, axes)

        axes.autoscale_view(scalex=False)
        axes.set_ylim([0, axes.get_ylim()[1]])  # assure the radial axis of a polar plot makes sense and starts at 0
        plt.title(self.plot_params['title'], color=self.plot_params['text_color'])
        plt.show()
        if self.plot_params['save_plot']:
            name = self.plot_params['title'].replace(' ', '_')
            name = name.replace('°', '')
            fig.savefig(name, dpi=300, facecolor=self.plot_params['face_color'])

    def writeFrames(self, direct_r=[6, 16, 26, 36, 46], ghost_r=[6, 26, 36], start=100, end=180, stepsize=10):
        steps = np.linspace(start, end, 1 + (end - start) // stepsize)
        print(steps)
        for a in tqdm(steps, position=0, desc='Writing frames'):
            self.setInclination(a)
            bh.plot_params['title'] = 'inclination = {:03}°'.format(int(a))
            bh.plotIsoradial(direct_r, ghost_r)


if __name__ == '__main__':
    bh = BlackHole()
    bh.writeFrames()
