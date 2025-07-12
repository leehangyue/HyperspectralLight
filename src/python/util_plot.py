import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.integrate import trapezoid
from util_led_spectrum import HyperspectralLight
from util_color_system import cs_p3 as cs


def plot_results(h_light: HyperspectralLight, target_spectrum: np.ndarray, output_spectrum: np.ndarray, 
                 wavelengths: np.ndarray,
                 exposure_adj_sum: float = 0.7, exposure_adj_led: float = 0.3, 
                 target_spectrum_blurred: np.ndarray = None, output_spectrum_blurred: np.ndarray = None,
                 channel_flux_ratios: np.ndarray = None):
    rgb_channels = []
    spec_ch = []
    for i in range(h_light.N_channels):
        spec = h_light.get_channel_spectrum(id_channel=i, flux_ratio=max(1e-9, channel_flux_ratios[i]))[1]
        rgb = cs.spec_to_rgb(wavelengths=h_light._wavelengths, spectrum=spec, out_fmt='html', 
                            exposure_adj=exposure_adj_led, brightness_wavelength_range=(340, 1020))
        rgb_channels.append(rgb)
        spec_ch.append(spec)
        rgb_leds = []
    for i in range(h_light.N_leds):
        spec = h_light.get_LED_spectrum(id_LED=i, flux_ratio=0.5)[1]
        rgb = cs.spec_to_rgb(wavelengths=h_light._wavelengths, spectrum=spec, out_fmt='html', 
                            exposure_adj=exposure_adj_led, brightness_wavelength_range=(340, 1020))
        rgb_leds.append(rgb)

    fig, axs = plt.subplots(ncols=2, nrows=3, constrained_layout=True, figsize=(8, 8))
    ax0, ax1, ax2, ax3, ax4, ax5 = axs.flatten()

    brightness_wavelength_range = (390, 710)
    ax0.set_facecolor(cs.spec_to_rgb(wavelengths=h_light._wavelengths, spectrum=target_spectrum, out_fmt='html', 
                                    exposure_adj=exposure_adj_sum, brightness_wavelength_range=brightness_wavelength_range))
    target_xyz = cs.spec_to_xyz(wavelengths=h_light._wavelengths, spectrum=target_spectrum)
    output_xyz = cs.spec_to_xyz(wavelengths=h_light._wavelengths, spectrum=output_spectrum)
    exposure_compensation = np.log(target_xyz[1] / output_xyz[1]) / np.log(2)
    # exposure_compensation = 0  # turn off exposure compensation
    ax0.add_patch(Rectangle(xy=(0.25, 0.25), width=0.5, height=0.5, edgecolor='k', linewidth=1, linestyle=(0, (1, 50)),
                            facecolor=cs.spec_to_rgb(wavelengths=h_light._wavelengths, spectrum=output_spectrum, out_fmt='html', 
                                                    exposure_adj=exposure_adj_sum+exposure_compensation, 
                                                    brightness_wavelength_range=brightness_wavelength_range)))
    ax0.annotate('Output color', color='k', xy=(0.5, 0.5), va='center', ha='center')
    ax0.set_title('Target color')

    c1 = '#3333aa'
    c2 = '#00aaff'
    c3 = '#dd6633'
    ax1.plot(wavelengths, target_spectrum, label='Target', c=c1, linewidth=0.6)
    ax1.plot(wavelengths, output_spectrum, label='Output', c=c2, linewidth=0.6)
    ax1.plot(wavelengths, target_spectrum_blurred, label='Target blurred', c=c1, linestyle='dashed')
    ax1.plot(wavelengths, output_spectrum_blurred, label='Output blurred', c=c2, linestyle='dashed')
    ax1_2 = ax1.twinx()
    ax1_2.bar(h_light.get_channel_nom_wls(), channel_flux_ratios, width=5, color=c3, alpha=0.3)
    ax1.grid('both')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Relative intensity (a.u.)', c=c1)
    ax1.set_ylim(bottom=0.)
    ax1_2.set_ylim(bottom=0.)
    ax1_2.set_ylabel('LED flux ratio', c=c3)
    ax1.legend()
    ax1.set_title('Spectrum simulator performance')

    # ax = ax2
    # led_peak_ids = np.array([np.argmax(led.spectrum[1:, :], axis=1) for led in h_light.get_LED_list()])
    # led_peak_wls = wavelengths[led_peak_ids]
    # led_fluxes = np.array([trapezoid(x=led.spectrum[0, :], y=led.spectrum[1:, :], axis=1) for led in h_light.get_LED_list()])
    # for i in range(h_light.N_leds):
    #     ax.plot(led_peak_wls[i], led_fluxes[i], c=rgb_leds[i], linewidth=1)
    #     ax.plot([h_light.led_nominal_wls[i], led_peak_wls[i, -1]], [0.5 * np.min(led_fluxes), led_fluxes[i][-1]], 
    #             c=rgb_leds[i], linestyle='dashed', linewidth=0.5)
    # ax.set_xlabel('Wavelength (nm)')
    # ax.set_ylabel('LED flux (W * m^-2) @ ref. pos.')
    # ax.set_yscale('log')
    # ax.grid('both')
    # ax.set_title('LED flux vs. peak wavelength')

    ax = ax3
    channel_spectra = np.array([[h_light.get_channel_spectrum(i, x)[1] 
                                 for x in np.power(10., np.linspace(0., -2., 7))] 
                                for i in range(h_light.N_channels)])
    channel_peak_ids = np.argmax(channel_spectra, axis=2)
    channel_peak_wls = wavelengths[channel_peak_ids]
    channel_fluxes = trapezoid(x=wavelengths, y=channel_spectra, axis=2)
    channel_nom_wls = h_light.get_channel_nom_wls()
    for i in range(h_light.N_channels):
        ax.plot(channel_peak_wls[i], channel_fluxes[i], c=rgb_channels[i], linewidth=1)
        ax.plot([channel_nom_wls[i], channel_peak_wls[i, -1]], [0.5 * np.min(channel_fluxes), channel_fluxes[i][-1]], 
                c=rgb_channels[i], linestyle='dashed', linewidth=0.5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Channel flux (W * m^-2) @ ref. pos.')
    ax.set_yscale('log')
    ax.grid('both')
    ax.set_title('Channel flux vs. peak wavelength')

    ax = ax4
    for i in range(h_light.N_channels):
        ax.plot(h_light._wavelengths, spec_ch[i], c=rgb_channels[i], linewidth=0.5)
        ax.fill_between(x=h_light._wavelengths, y1=spec_ch[i], y2=0, facecolor=rgb_channels[i], alpha=0.1)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Spectral intensity (W * m^-2 * nm^-1)')
    ax.grid('both')
    ax.set_title('Absolute channel spectra')

    ax = ax5
    for i in range(h_light.N_channels):
        spec_norm = spec_ch[i] / (1e-308 + np.max(spec_ch[i]))
        ax.plot(h_light._wavelengths, spec_norm, c=rgb_channels[i], linewidth=0.5)
        ax.fill_between(x=h_light._wavelengths, y1=spec_norm, y2=0, facecolor=rgb_channels[i], alpha=0.1)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Normalized spectral intensity (a.u.)')
    ax.grid('both')
    ax.set_title('Relative channel spectra')

    return fig, axs


def show_plot():
    plt.show()