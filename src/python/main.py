import numpy as np
from util_led_spectrum import HyperspectralLight, read_spectrum_file, get_led_list_from_fit_results
from util_color_system import blackbody_radiation, cherenkov_radiation
from util_pca9685 import PCA9685
from time import sleep, time_ns
from os.path import join, dirname
# from res.LEDtestdata.fit_model import fit_model_pars, calc_channel_data, get_default_channel_list_from_fit_results


def main_static():
    
    # flux_ratio_min = 5e-5
    # flux_ratio_max = 0.5  # with values greater than 0.5, the rectifier diodes may heat up significantly

    # R_wirings = 0.5  # Ohm
    # I_nominal = 0.7  # Amperes, deprecated

    # I_max = 0.7  # Amperes, deprecated

    # # include_plain_diodes = True
    # include_plain_diodes = False

    # if include_plain_diodes:
    #     V_low_lim = 1.6
    # else:
    #     V_low_lim = 0.

    # fname_fit_results = fit_model_pars(mute=True)
    # led_list = get_led_list_from_fit_results(fname_led_fit_results=fname_fit_results, I_nominal=I_nominal)
    # led_tags = [led.tag for led in led_list]
    # channel_list = get_default_channel_list_from_fit_results(fname_fit_results=fname_fit_results)
    # # get the nominal wavelength of each led in led_list
    # nom_wls_led = np.array([led.nominal_wavelength for led in led_list])
    # # get the nominal wavelength of the first component in each channel in channel_list
    # nom_wls_channel = np.array([led_list[led_tags.index(list(ch.keys())[0])].nominal_wavelength for ch in channel_list])

    # wl_to_operate = 560
    # channel_list[np.argmin(np.abs(nom_wls_channel - wl_to_operate))]\
    #     [led_list[np.argmin(np.abs(nom_wls_led - wl_to_operate))].tag] = 4  # change channel led count

    # # delete the 575 nm Ruixiang channel
    # channel_main_led_tags = [list(ch.keys())[0] for ch in channel_list]
    # del channel_list[channel_main_led_tags.index('575 Ruixiang 2024-02-28')]
    # # get the nominal wavelength of the first component in each channel in channel_list
    # nom_wls_channel = np.array([led_list[led_tags.index(list(ch.keys())[0])].nominal_wavelength for ch in channel_list])

    # channel_main_led_tags = [list(ch.keys())[0] for ch in channel_list]
    # channel_list[channel_main_led_tags.index('575 Baideqi 2024-03-19')]\
    #     [led_list[np.argmin(np.abs(nom_wls_led - 575))].tag] = 4  # change channel led count

    # wl_to_operate = 590
    # channel_list[np.argmin(np.abs(nom_wls_channel - wl_to_operate))]\
    #     [led_list[np.argmin(np.abs(nom_wls_led - wl_to_operate))].tag] = 4  # change channel led count

    # # merge the 365 nm channel into the 380 nm channel
    # channel_list[np.argmin(np.abs(nom_wls_channel - 380))] |= channel_list[np.argmin(np.abs(nom_wls_channel - 365))]
    # del channel_list[np.argmin(np.abs(nom_wls_channel - 365))]
    # # update the nominal wavelength of the first component in each channel in channel_list
    # nom_wls_channel = np.array([led_list[led_tags.index(list(ch.keys())[0])].nominal_wavelength for ch in channel_list])

    # # merge the 940 nm Taiyi 2024-03-19 channel into the 940 nm Jingzhong 2024-02-28 channel
    # channel_main_led_tags = [list(ch.keys())[0] for ch in channel_list]
    # id_ch_to_del = channel_main_led_tags.index('940 Taiyi 2024-03-19')
    # channel_list[channel_main_led_tags.index('940 Jingzhong 2024-02-28')] |= channel_list[id_ch_to_del]
    # del channel_list[id_ch_to_del]
    # # update the nominal wavelength of the first component in each channel in channel_list
    # nom_wls_channel = np.array([led_list[led_tags.index(list(ch.keys())[0])].nominal_wavelength for ch in channel_list])

    # # delete the 760 nm channel
    # del channel_list[np.argmin(np.abs(nom_wls_channel - 760))]
    # # get the nominal wavelength of the first component in each channel in channel_list
    # nom_wls_channel = np.array([led_list[led_tags.index(list(ch.keys())[0])].nominal_wavelength for ch in channel_list])

    # # # delete UV channels
    # # for i, wl in enumerate(reversed(nom_wls_channel)):
    # #     if wl <= 400:
    # #         del channel_list[np.argmin(np.abs(nom_wls_channel - wl))]
    # # # get the nominal wavelength of the first component in each channel in channel_list
    # # nom_wls_channel = np.array([led_list[led_tags.index(list(ch.keys())[0])].nominal_wavelength for ch in channel_list])

    # # # delete IR channels
    # # for i, wl in enumerate(reversed(nom_wls_channel)):
    # #     if wl >= 630:
    # #         del channel_list[np.argmin(np.abs(nom_wls_channel - wl))]
    # # # get the nominal wavelength of the first component in each channel in channel_list
    # # nom_wls_channel = np.array([led_list[led_tags.index(list(ch.keys())[0])].nominal_wavelength for ch in channel_list])

    # fname_channel_data = calc_channel_data(channel_list=channel_list, 
    #                                        led_list=led_list, 
    #                                        flux_ratio_min=flux_ratio_min, flux_ratio_max=flux_ratio_max, 
    #                                        R_wirings=R_wirings, V_low_lim=V_low_lim)
    h_light = HyperspectralLight(min_wl=340, max_wl=1020, 
                                #  channel_spectra_path=join(dirname(dirname(dirname(__file__))), "res/channel_calibration_data_open"),
                                 channel_spectra_path=join(dirname(dirname(dirname(__file__))), "res/channel_calibration_data_barrel"),
                                 )
    wavelengths = h_light._wavelengths

    correction_coefs = np.array(h_light.get_channel_nom_wls()) / 1e3
    correction_coefs = -correction_coefs + 1.6
    correction_coefs /= np.nanmean(correction_coefs)
    for i in range(h_light.N_channels):
        correction = correction_coefs[i]
        if correction > 0:  # is not NaN
            h_light._channel_list[i].spectrum_lib *= correction
    # h_light._channel_list[11].spectrum_lib *= 1.1

    blur_radius = 15  # nm
    blur_kernel = np.exp(-((np.arange(-blur_radius * 3, blur_radius * 3 + 1)) / blur_radius) ** 2 / 2)  # gaussian kernel
    blur_kernel /= np.sum(blur_kernel)

    uv_off = True
    # uv_off = False

    ir_off = True
    # ir_off = False

    max_flux_ratio = 0.9  # maximum flux ratio for any channel

    off_channel_mask = np.array([False] * h_light.N_channels)
    # off_channel_mask[1::2] = True

    # # measure or generate target
    calibration_file = 'res/LEDtestdata/multiply_this_with_spectra_before_202403150108.csv'
    target_spectrum = blackbody_radiation(wavelength=wavelengths, temperature=5500, normalize=True)
    # target_spectrum = cherenkov_radiation(wavelength=wavelengths)
    # target_spectrum = wavelengths * 0 + 1  # equi-power
    # target_spectrum = read_spectrum_file(fname='res/sample spectra/SunsetOrdinary/merged_spectra_only.csv', wavelengths=wavelengths, intensity_column=150)
    # target_spectrum = read_spectrum_file(fname='res/sample spectra/SunsetRed/merged_spectra_only.csv', wavelengths=wavelengths, intensity_column=120)
    # target_spectrum = read_spectrum_file(fname='res/sample spectra/cloudy_daylight_20240307131758.csv', wavelengths=wavelengths, fname_calibration=calibration_file)
    # target_spectrum = read_spectrum_file(fname='res/sample spectra/direct_sunlight_20240312123009.csv', wavelengths=wavelengths, fname_calibration=calibration_file)
    # target_spectrum = read_spectrum_file(fname='res/sample spectra/gold_fluorescent_light_20240225194221.csv', wavelengths=wavelengths, fname_calibration=calibration_file)
    # target_spectrum = read_spectrum_file(fname='res/sample spectra/green_leaf_under_cloudy_daylight_20240225160513.csv', wavelengths=wavelengths, fname_calibration=calibration_file)
    # target_spectrum = read_spectrum_file(fname='res/sample spectra/high_pressure_sodium_light_20240225193613.csv', wavelengths=wavelengths, fname_calibration=calibration_file)
    # target_spectrum = read_spectrum_file(fname='res/sample spectra/peanut_oil_under_cloudy_daylight_20240307131813.csv', wavelengths=wavelengths, fname_calibration=calibration_file)
    # target_spectrum = read_spectrum_file(fname='res/sample spectra/rapseed_oil_under_cloudy_daylight_20240307131622.csv', wavelengths=wavelengths, fname_calibration=calibration_file)
    # target_spectrum = read_spectrum_file(fname='res/sample spectra/warm_LED_light_20240314182106.csv', wavelengths=wavelengths, fname_calibration=calibration_file)
    # target_spectrum = read_spectrum_file(fname='res/sample spectra/white_fluorescent_light_20240225193438.csv', wavelengths=wavelengths, fname_calibration=calibration_file)
    # target_spectrum = read_spectrum_file(fname='res/sample spectra/daylight_thru_napkin_dragonfruit_20240315131135.csv', wavelengths=wavelengths)
    # target_spectrum *= read_spectrum_file(fname='res/sample spectra/dragon_fruit_pink_transmittance.csv', wavelengths=wavelengths)
    # target_spectrum *= read_spectrum_file(fname='res/sample spectra/QB19_transmittance_resampled.csv', wavelengths=wavelengths)
    # target_spectrum *= read_spectrum_file(fname='res/sample spectra/QB21_transmittance_resampled.csv', wavelengths=wavelengths)
    # target_spectrum *= read_spectrum_file(fname='res/sample spectra/CB590_transmittance.csv', wavelengths=wavelengths)

    target_spectrum_blurred = np.convolve(target_spectrum, blur_kernel, mode='same')
    intensity_adjusting_factor = 1. / np.max(target_spectrum_blurred)
    target_spectrum *= intensity_adjusting_factor
    target_spectrum_blurred *= intensity_adjusting_factor

    # fit spectrum
    if uv_off:
        h_light.min_wl = 410
        off_channel_mask[:2] = True
    else:
        h_light.min_wl = 370
    if ir_off:
        h_light.max_wl = 720
        off_channel_mask[-10:] = True
    else:
        h_light.max_wl = 990

    channel_flux_ratios = h_light.calc_channel_flux_ratios(
        wavelengths=wavelengths, target_spectrum=target_spectrum, blur_radius=5, 
        max_flux_ratio=max_flux_ratio, off_channel_mask=off_channel_mask)
    h_light.min_wl = 340
    h_light.max_wl = 1020
    _, output_spectrum = h_light.output_spectrum(channel_flux_ratios=channel_flux_ratios)
    output_spectrum_blurred = np.convolve(output_spectrum, blur_kernel, mode='same')
    intensity_adjusting_factor = 1. / np.max(output_spectrum_blurred)
    output_spectrum *= intensity_adjusting_factor
    output_spectrum_blurred *= intensity_adjusting_factor

    # # show plot before applying PWM ratios
    # from util_plot import plot_results, show_plot
    # fig, axs = plot_results(h_light=h_light, target_spectrum=target_spectrum, output_spectrum=output_spectrum, 
    #                         wavelengths=wavelengths,
    #                         target_spectrum_blurred=target_spectrum_blurred, output_spectrum_blurred=output_spectrum_blurred,
    #                         channel_flux_ratios=channel_flux_ratios)
    # show_plot()

    pwm_ratios = h_light.get_pwm_ratios_from_channel_flux_ratios(channel_flux_ratios=channel_flux_ratios)
    pwm_ratios = np.minimum(1., pwm_ratios)

    # pwm_ratios[:12] = 0.
    # pwm_ratios[13:] = 0.
    # pwm_ratios *= 0. / np.max(pwm_ratios[:15])
    # pwm_ratios *= 1. / np.max(pwm_ratios[15:])
    # pwm_ratios = np.ones(30) * 1.
    
    # pwm_ratios = np.zeros(30)
    # pwm_ratios[2] = 0.1

    print(f"Number of channels: {len(pwm_ratios)}")
    print(f"Maximum channel flux ratio: {np.max(pwm_ratios):.3f}")
    print("Channel flux ratios:")
    print(", ".join([f"{x:.3f}" for x in pwm_ratios[:15]]))
    print(", ".join([f"{x:.3f}" for x in pwm_ratios[15:]]))

    ctrl1_channels = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
    ctrl2_channels = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29]

    pca9685 = PCA9685()

    pwm_ratios_1 = np.append(pwm_ratios[ctrl1_channels], 0.)
    pca9685.i2c_address = int(0x43)
    pca9685.set_channels(channel_flux_ratios=pwm_ratios_1, offset=0.)

    pwm_ratios_2 = np.append(pwm_ratios[ctrl2_channels], 0.)
    pca9685.i2c_address = int(0x42)
    pca9685.set_channels(channel_flux_ratios=pwm_ratios_2, offset=0.)
    # sleep(0.8)

    # pwm_ratios = np.zeros(30)

    # pwm_ratios_1 = np.append(pwm_ratios[ctrl1_channels], 0.)
    # pca9685.i2c_address = int(0x43)
    # pca9685.set_channels(channel_flux_ratios=pwm_ratios_1)

    # pwm_ratios_2 = np.append(pwm_ratios[ctrl2_channels], 0.)
    # pca9685.i2c_address = int(0x42)
    # pca9685.set_channels(channel_flux_ratios=pwm_ratios_2)
    # # sleep(1.2)

    pca9685.sender.stop()


def main_animate(fps: float = 0.5, time_resolution_ms = 10.):
    interval = 1. / fps
    h_light = HyperspectralLight(min_wl=340, max_wl=1020, 
                                #  channel_spectra_path=join(dirname(dirname(dirname(__file__))), "res/channel_calibration_data_open"),
                                 channel_spectra_path=join(dirname(dirname(dirname(__file__))), "res/channel_calibration_data_barrel"),
                                 )
    wavelengths = h_light._wavelengths

    correction_coefs = np.array(h_light.get_channel_nom_wls()) / 1e3
    correction_coefs = -correction_coefs + 1.6
    correction_coefs /= np.nanmean(correction_coefs)
    for i in range(h_light.N_channels):
        correction = correction_coefs[i]
        if correction > 0:  # is not NaN
            h_light._channel_list[i].spectrum_lib *= correction
    # h_light._channel_list[11].spectrum_lib *= 1.1

    blur_radius = 15  # nm
    blur_kernel = np.exp(-((np.arange(-blur_radius * 3, blur_radius * 3 + 1)) / blur_radius) ** 2 / 2)  # gaussian kernel
    blur_kernel /= np.sum(blur_kernel)

    uv_off = True
    # uv_off = False

    ir_off = True
    # ir_off = False

    max_flux_ratio = 0.9  # maximum flux ratio for any channel

    off_channel_mask = np.array([False] * h_light.N_channels)
    # off_channel_mask[1::2] = True

    ctrl1_channels = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
    ctrl2_channels = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29]

    # from zenodo_dataset_loader import ZenodoDataset
    # zenodo_dataset = ZenodoDataset()
    # spectrum_list = zenodo_dataset.load("daylight_timelapse")

    spectrum_list = [
        {
            "wavelengths": wavelengths,
            "spectrum": read_spectrum_file(fname='res/sample spectra/SunsetRed/merged_spectra_only.csv', 
                                           wavelengths=wavelengths, 
                                           intensity_column=i)
        }
        for i in range(190)
    ]

    for spectrum_dict in spectrum_list:
        time0 = time_ns()
        sample_wls = spectrum_dict["wavelengths"]
        sample_spt = spectrum_dict["spectrum"]
        target_spectrum = np.interp(
            x=wavelengths, 
            xp=sample_wls,
            fp=sample_spt,
            left=0., right=0.,
        )

        target_spectrum_blurred = np.convolve(target_spectrum, blur_kernel, mode='same')
        intensity_adjusting_factor = 1. / np.max(target_spectrum_blurred)
        target_spectrum *= intensity_adjusting_factor
        target_spectrum_blurred *= intensity_adjusting_factor

        # fit spectrum
        if uv_off:
            h_light.min_wl = 410
            off_channel_mask[:2] = True
        else:
            h_light.min_wl = 370
        if ir_off:
            h_light.max_wl = 720
            off_channel_mask[-10:] = True
        else:
            h_light.max_wl = 990

        channel_flux_ratios = h_light.calc_channel_flux_ratios(
            wavelengths=wavelengths, target_spectrum=target_spectrum, blur_radius=5, 
            max_flux_ratio=max_flux_ratio, off_channel_mask=off_channel_mask)
        h_light.min_wl = 340
        h_light.max_wl = 1020
        _, output_spectrum = h_light.output_spectrum(channel_flux_ratios=channel_flux_ratios)
        output_spectrum_blurred = np.convolve(output_spectrum, blur_kernel, mode='same')
        intensity_adjusting_factor = 1. / np.max(output_spectrum_blurred)
        output_spectrum *= intensity_adjusting_factor
        output_spectrum_blurred *= intensity_adjusting_factor

        pwm_ratios = h_light.get_pwm_ratios_from_channel_flux_ratios(channel_flux_ratios=channel_flux_ratios)
        pwm_ratios = np.minimum(1., pwm_ratios)

        pca9685 = PCA9685()

        pwm_ratios_1 = np.append(pwm_ratios[ctrl1_channels], 0.)
        pca9685.i2c_address = int(0x43)
        pca9685.set_channels(channel_flux_ratios=pwm_ratios_1, offset=0.)

        pwm_ratios_2 = np.append(pwm_ratios[ctrl2_channels], 0.)
        pca9685.i2c_address = int(0x42)
        pca9685.set_channels(channel_flux_ratios=pwm_ratios_2, offset=0.)

        pca9685.sender.stop()

        time1 = time_ns()
        while time1 - time0 < interval * 1e9:
            sleep(time_resolution_ms * 1e-3)
            time1 = time_ns()


if __name__ == "__main__":
    try:
        # main_static()
        main_animate()
    finally:
        from util_pca9685 import lights_off
        lights_off()
    