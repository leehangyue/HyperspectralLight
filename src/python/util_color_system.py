import numpy as np

def blackbody_radiation(wavelength, temperature, normalize:bool=False):
    """
    Calculates the blackbody radiation spectrum at given temperature
    Reference: https://phys.libretexts.org/Bookshelves/University_Physics/University_Physics_(OpenStax)/University_Physics_III_-_Optics_and_Modern_Physics_(OpenStax)/06%3A_Photons_and_Matter_Waves/6.02%3A_Blackbody_Radiation
    
    Arguments:
        wavelength : float or ndarray
            nanometers (nm)
        temperature : float
            Kelvin (K)
    Returns:
        float or ndarray of the same shape as wavelength
        blackbody radiation spectrum at given temperature, W * m^-2 * nm^-1
    """
    # kB = 1.38064852e-23  # m^2 * kg * s^-2 * K^-1 or J * K^-1  # Boltzmann constant
    # h = 6.62607015e-34  # J * s  # Planck's constant
    # c = 299792458. # m * s^-1  # the speed of light in vacuum
    from scipy.constants import k as kB
    from scipy.constants import h, c
    wavelength_m = wavelength * 1e-9  # wavelength in meters
    intensity = 2 * h * c ** 2 / (wavelength_m ** 5 * (np.exp(h * c / (wavelength_m * kB * temperature)) - 1))  # W * m^-2(area) * m^-1(wavelength)
    if normalize:
        intensity /= np.max(intensity)
    else:
        intensity *= 1e-9
    return intensity

def cherenkov_radiation(wavelength):
    """Relative spectrum of cherenkov radiation"""
    intensity = 1 / wavelength ** 2
    return intensity / np.max(intensity)

def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1-x-y))

def XYZ_from_xyY(x, y, Y):
    xyz = xyz_from_xy(x, y)
    return xyz * Y / y

def gamma(rgb):
    """convert RGB to linear values"""
    # https://en.wikipedia.org/wiki/SRGB
    rgb = np.array(rgb)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

def invgamma(linear):
    """convert linear values to RGB"""
    linear = np.array(linear)
    return np.where(linear <= 0.0031308, linear * 12.92, linear ** (1 / 2.4) * 1.055 - 0.055)

# the method cmf is based on: https://en.wikipedia.org/wiki/CIE_1931_color_space
def cmf(wavelengths):
    """The CIE color matching function on given grid points of wavelengths"""
    def g(wl, mu, tau1, tau2):
        res = np.where(wl < mu, np.exp(-(tau1 * (wl - mu)) ** 2 / 2), np.exp(-(tau2 * (wl - mu)) ** 2 / 2))
        return res
    x = 1.056 * g(wavelengths, 599.8, 0.0264, 0.0323) + \
        0.362 * g(wavelengths, 442.0, 0.0624, 0.0374) - \
        0.065 * g(wavelengths, 501.1, 0.0490, 0.0382)
    y = 0.821 * g(wavelengths, 568.8, 0.0213, 0.0247) + \
        0.286 * g(wavelengths, 530.9, 0.0613, 0.0322)
    z = 1.217 * g(wavelengths, 437.0, 0.0845, 0.0278) + \
        0.681 * g(wavelengths, 459.0, 0.0385, 0.0725)
    return np.vstack([x, y, z]).T

class ColorSystem:
    """A class representing a color system.

    A color system defined by the CIE x, y and z=1-x-y coordinates of
    its three primary illuminants and its "white point".

    The class ColorSystem is based on: https://scipython.com/blog/converting-a-spectrum-to-a-colour/

    """

    def __init__(self, red, green, blue, white):
        """Initialise the ColorSystem object.

        Pass vectors (i.e. NumPy arrays of shape (3,)) for each of the
        red, green, blue  chromaticities and the white illuminant
        defining the color system.

        """

        # Chromaticities
        self.red, self.green, self.blue = red, green, blue
        self.white = white
        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T
        self.MI = np.linalg.inv(self.M)
        # White scaling array
        self.wscale = self.MI.dot(self.white)
        # xyz -> rgb transformation matrix
        self.T = self.MI / self.wscale[:, np.newaxis]

    def xyz_to_rgb(self, xyz, out_fmt=None, brightness:float=1.0):
        """Transform from xyz to rgb representation of color.

        The output rgb components are normalized on their maximum
        value. If xyz is out the rgb gamut, it is desaturated until it
        comes into gamut.

        By default, fractional rgb components are returned; if
        out_fmt='html', the HTML hex string '#rrggbb' is returned.

        """

        rgb = self.T.dot(xyz)
        if np.any(rgb < 0):
            # if not in the RGB gamut: approximate by desaturating
            rgb -= np.min(rgb)
        if np.max(rgb) > 0.:
            # Normalize the rgb vector
            rgb /= np.max(rgb)

        rgb = rgb * brightness
        rgb = np.tanh(rgb)
        # rgb = invgamma(rgb)

        if out_fmt == 'html':
            return self.rgb_to_hex(rgb)
        return rgb

    def rgb_to_hex(self, rgb):
        """Convert from fractional rgb values to HTML-style hex string."""
        hex_rgb = (255 * rgb).astype(int)
        return '#{:02x}{:02x}{:02x}'.format(*hex_rgb)

    def spec_to_xyz(self, wavelengths, spectrum):
        """Convert a spectrum to an xyz point.

        Generate the color-matching functions on the same grid
          of wavelength points of the input spectrum

        """

        cmf_num = cmf(wavelengths)
        XYZ = spectrum @ cmf_num
        return XYZ
        # den = np.sum(spectrum) * np.sum(cmf_num)
        # if den == 0.:
        #     return XYZ
        # return XYZ / den

    def spec_to_rgb(self, wavelengths, spectrum, out_fmt=None, exposure_adj:float=0.0, 
                    brightness_wavelength_range:tuple=(380, 780)):
        """Convert a spectrum to an rgb value."""
        xyz = self.spec_to_xyz(wavelengths, spectrum)
        brightness_wavelength_mask = np.logical_and(wavelengths >= brightness_wavelength_range[0], 
                                                    wavelengths <= brightness_wavelength_range[1])
        brightness = np.sum(self.spec_to_xyz(wavelengths[brightness_wavelength_mask], 
                                             spectrum[brightness_wavelength_mask]))\
                     / (1e-308 + np.sum(spectrum[brightness_wavelength_mask])) * (2 ** exposure_adj)
        return self.xyz_to_rgb(xyz, out_fmt, brightness=brightness)


def plot_spectrum(wavelengths, spectra, exposure_adj_sum=0.7, brightness_wavelength_range=(390, 710)):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from tqdm import tqdm
    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True, figsize=(8, 1))
    n_spec = len(spectra)
    print("Visualizing...")
    for i in tqdm(range(n_spec)):
        ax.add_patch(Rectangle(xy=(i/n_spec, 0.), width=1/(n_spec), height=1., 
                               edgecolor='none', linewidth=1, linestyle=(0, (1, 50)),
                            facecolor=cs_p3.spec_to_rgb(wavelengths=wavelengths, spectrum=spectra[i], out_fmt='html', 
                                                        exposure_adj=exposure_adj_sum, 
                                                        brightness_wavelength_range=brightness_wavelength_range)))
    plt.show()


def plot_spectrum_from_file(fname):
    wavelengths = np.arange(340, 1020.1, 1)
    with open(fname, 'r') as f:
        while True:
            try:
                line = f.readline()
            except UnicodeDecodeError:
                continue
            n_col = len(line.split(','))
            if n_col > 1:
                break
    from util_led_spectrum import read_spectrum_file
    from tqdm import tqdm
    print("Reading spectrum file...")
    spectra = np.array([
        read_spectrum_file(fname=fname, wavelengths=wavelengths, intensity_column=i)
        for i in tqdm(range(1, n_col))
    ])
    plot_spectrum(wavelengths=wavelengths, spectra=spectra)


wl = np.arange(380, 780.5, 1)
cmf_num = cmf(wl)
white = blackbody_radiation(wl, 6500, True) @ cmf_num
white /= np.sum(white)

# https://en.wikipedia.org/wiki/SRGB
cs_srgb = ColorSystem(red=xyz_from_xy(0.64, 0.33),
                      green=xyz_from_xy(0.3, 0.6), 
                      blue=xyz_from_xy(0.15, 0.06), 
                      white=white)

# https://en.wikipedia.org/wiki/DCI-P3
cs_p3 = ColorSystem(red=xyz_from_xy(0.680, 0.320),
                    green=xyz_from_xy(0.365, 0.690), 
                    blue=xyz_from_xy(0.150, 0.060), 
                    white=white)

if __name__ == '__main__':
    # wl = np.arange(340, 1020.5, 1)
    # cmf_num = cmf(wl)
    # print(cs_srgb.spec_to_rgb(
    #     wavelengths=wl, spectrum=blackbody_radiation(wl, 6500, True),
    #     out_fmt=None, exposure_adj=0.))

    # import matplotlib.pyplot as plt
    # colors = ['r', 'g', 'b']
    # for i in range(3):
    #     plt.plot(wl, cmf_num[:, i], c=colors[i])
    # plt.grid('both')
    # plt.show()

    plot_spectrum_from_file(fname='sample spectra/SunsetRed/merged_spectra_only.csv')
    # plot_spectrum_from_file(fname='sample spectra/green_leaf_under_cloudy_daylight_20240225160513.csv')
