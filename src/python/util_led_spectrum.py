import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from os import listdir
from os.path import split, splitext, join, exists, abspath, getmtime
from typing import Optional, Union

class LED:
    """LED information including spectrum and voltage-current-flux relation.

    Attributes
    ----------
    tag : str
        LED identifier including vendor name, distributor name, purchase date etc.
    nominal_wavelength : float
        Nominal wavelength in nanometers (nm)
    I_nominal : float
        Nominal operating current in Amperes (A)
    spectrum : np.ndarray
        (1+N_levels) x N_wavelengths array where:
        - Row 0: Sampling wavelengths (nm)
        - Row 1: Measured intensities (W*m^-2*nm^-1) at max level
        - Rows 2+: (Optional) Measured intensities at dimmed levels
    fit_data : dict
        Voltage-flux relation parameters containing:
        - 'etakBT_e' (V): Thermal voltage
        - 'V_ref' (V): Reference voltage  
        - 'V_loss' (V): Series resistance voltage drop
        - 'satuation' (1): Current saturation parameter

    Methods
    -------
    get_voltage_from_flux(flux_ratio: float) -> float
        Calculate voltage needed for given flux ratio
    get_flux_from_voltage(voltage: float) -> float  
        Calculate flux ratio from applied voltage
    get_current_from_flux(flux_ratio: float) -> float
        Calculate current ratio for given flux ratio
    get_flux_from_current(I_ratio: float) -> float
        Calculate flux ratio from current ratio
    get_wavelengths() -> np.ndarray
        Get wavelength sampling points
    get_spectrum(flux_ratio: float, wavelengths: Optional[np.ndarray]) -> np.ndarray
        Get spectrum at given flux ratio, optionally resampled
    """
    def __init__(self, tag:str=None, nominal_wavelength:float=None, I_nominal:float=None) -> None:
        self.tag = tag
        self.nominal_wavelength = nominal_wavelength
        self.I_nominal = I_nominal
        self.spectrum = None
        self.fit_data = None

    def get_voltage_from_flux(self, flux_ratio: float) -> float:
        """Calculate voltage needed to achieve given flux ratio.
        
        Parameters
        ----------
        flux_ratio : float
            Desired flux ratio (0-1) relative to nominal
        
        Returns
        -------
        float
            Required voltage in Volts
            
        Notes
        -----
        Uses diode equation with parameters from fit_data:
        V = etakBT_e*ln(I_ratio + Is_Inom) + V_ref + I_ratio*V_loss
        """
        return flux2voltage(flux_ratio=flux_ratio, 
                            etakBT_e=self.fit_data['etakBT_e'], 
                            V_ref=self.fit_data['V_ref'], 
                            V_loss=self.fit_data['V_loss'], 
                            satuation=self.fit_data['satuation'])
    
    def get_flux_from_voltage(self, voltage: float) -> float:
        """Calculate flux ratio resulting from applied voltage.
        
        Parameters
        ----------
        voltage : float
            Applied voltage in Volts
            
        Returns
        -------
        float
            Resulting flux ratio (0-1)
            
        Notes
        -----
        Numerically solves diode equation for flux ratio.
        May raise ValueError if voltage outside operating range.
        """
        return voltage2flux(voltage=voltage, 
                            etakBT_e=self.fit_data['etakBT_e'], 
                            V_ref=self.fit_data['V_ref'], 
                            V_loss=self.fit_data['V_loss'], 
                            satuation=self.fit_data['satuation'])
    
    def get_current_from_flux(self, flux_ratio: float) -> float:
        """Calculate current ratio needed for given flux ratio.
        
        Parameters
        ----------
        flux_ratio : float
            Desired flux ratio (0-1)
            
        Returns
        -------
        float
            Current ratio (I/I_nominal) needed
            
        Notes
        -----
        Uses saturation model:
        flux_ratio = (1 - exp(-satuation*I_ratio))/(1 - exp(-satuation))
        """
        return flux2current(flux_ratio=flux_ratio, 
                            satuation=self.fit_data['satuation'])
    
    def get_flux_from_current(self, I_ratio: float) -> float:
        """Calculate flux ratio resulting from current ratio.
        
        Parameters
        ----------
        I_ratio : float
            Current ratio (I/I_nominal)
            
        Returns
        -------
        float
            Resulting flux ratio (0-1)
            
        Notes
        -----
        Uses inverse of saturation model from get_current_from_flux().
        """
        return current2flux(I_ratio=I_ratio, 
                            satuation=self.fit_data['satuation'])
    
    def get_wavelengths(self) -> np.ndarray:
        """Get wavelength sampling points for spectrum measurements.
        
        Returns
        -------
        np.ndarray
            Array of wavelength values in nanometers
        """
        return self.spectrum[0, :]

    def get_spectrum(self, flux_ratio: float = 1., wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        """Get LED spectrum at given flux ratio, optionally resampled.
        
        Parameters
        ----------
        flux_ratio : float, optional
            Flux ratio (0-1), by default 1.0 (full power)
        wavelengths : Optional[np.ndarray], optional
            Target wavelengths for resampling, by default None (use native)
            
        Returns
        -------
        np.ndarray
            Spectral intensities in W*m^-2*nm^-1
            
        Notes
        -----
        - For flux_ratio < 1, interpolates between measured dim levels
        - For wavelengths=None, returns spectrum at native sampling
        - Clips flux_ratio to minimum 0
        """
        # if not (0 <= flux_ratio <= 1):
        #     raise ValueError('flux_ratio out of range! 0 <= flux_ratio <= 1, got flux_ratio = ' + str(flux_ratio))
        flux_ratio = max(0., flux_ratio)
        wavelengths_lib = self.spectrum[0, :]
        if flux_ratio < 1:
            led_spectrum = self.spectrum[1:, :]
            led_spectrum = np.vstack((led_spectrum, led_spectrum[0] * 0.))
            fluxes = trapezoid(y=led_spectrum, x=wavelengths_lib, axis=1)
            fluxes /= np.max(fluxes)
            f = interp1d(x=fluxes, y=led_spectrum, axis=0, bounds_error=False)
            led_spectrum = f(flux_ratio)
        else:
            led_spectrum = self.spectrum[1, :] * flux_ratio
        if wavelengths is not None:
            # resample
            f = interp1d(x=wavelengths_lib, y=led_spectrum, bounds_error=False, fill_value=0.)
            led_spectrum = f(wavelengths)
        return led_spectrum

class HyperspectralLight:
    """Tunable hyperspectral light source composed of multiple LED channels.
    
    Attributes
    ----------
    resolution : float
        Spectral sampling interval in nanometers (nm)
    min_wl : float
        Minimum wavelength of analyzed spectrum (nm)
    max_wl : float
        Maximum wavelength of analyzed spectrum (nm)
        May differ from set value to ensure (max_wl - min_wl) is integer multiple of resolution
    R_wirings : float or array
        Wiring resistance per channel in Ohms
    I_nominal : float
        Nominal operating current in Amperes (A)
    _led_list : list[LED]
        List of LED instances
    _channel_list : list[dict]
        List of channel configurations, each as {LED.tag: count}
    _channel_flux_ratios : Optional[np.ndarray]
        Buffer for calculated channel flux ratios
    _wavelengths : Optional[np.ndarray]
        Wavelength sampling points
    """

    def __init__(self, led_data_path: str = "./LEDtestdata/", 
                 resolution: float = 1., min_wl: float = 300, max_wl: float = 1200) -> None:
        """Initialize hyperspectral light source with LED configuration.
        
        Parameters
        ----------
        led_data_path : str, optional
            Path to LED test data directory, by default "./LEDtestdata/"
        resolution : float, optional
            Spectral sampling resolution in nm, by default 1.0
        min_wl : float, optional
            Minimum wavelength in nm, by default 300
        max_wl : float, optional
            Maximum wavelength in nm, by default 1200
            
        Notes
        -----
        Automatically loads LED spectrum data and fit parameters if led_data_path exists.
        Otherwise initializes with virtual LED data.
        """
        self.resolution = resolution
        self.min_wl = min_wl
        self.max_wl = max_wl
        self.R_wirings = 0.  # Ohm
        self.I_nominal = 0.7  # Ampere
        self._led_list = []
        self._channel_list = []
        self._channel_flux_ratios = None  # buffer, updates after calling calc_channel_flux_ratios
        self._wavelengths = None
        if led_data_path is not None and exists(abspath(led_data_path)):
            self._load_spectrum_data(pathname=led_data_path)
            # self._load_led_fit_pars(fname_fit_results=join(led_data_path, 'Summary_fit_results.csv'))
            self._load_channel_data(pathname=led_data_path)
        else:
            self._wavelengths = np.arange(self.min_wl, self.max_wl + self.resolution * 0.999, self.resolution)
            self._gen_leds_and_channels()

    @property
    def N_leds(self):
        return len(self.get_LED_list())
    
    @property
    def N_diodes(self):
        return len(self._led_list)
    
    @property
    def N_channels(self):
        return len(self._channel_list)
    
    @property
    def led_nominal_wls(self):
        return np.array([led.nominal_wavelength for led in self.get_LED_list()])
    
    @property
    def all_nominal_wls(self):
        return np.array([led.nominal_wavelength for led in self._led_list])
    
    @property
    def all_tags(self):
        return [led.tag for led in self._led_list]

    def _load_spectrum_data(self, pathname, wl_tol:float=1e-3, 
                            calibration_file:str='multiply_this_with_spectra_before_202403150108.csv'):
        raw_data_pathname = join(pathname, 'rawData')
        flist = listdir(raw_data_pathname)
        flist = [f for f in flist if (not f.startswith('.')) and splitext(f)[1].lower() == '.csv']
        flist.sort(reverse=True)
        self._led_list = []
        for f in flist:
            fname = join(raw_data_pathname, f)  # f: name.extension, fname:path/name.extension
            wls, intensities = read_spectrum_file(fname=fname, 
                                                  fname_calibration=join(pathname, calibration_file))
            
            if self._wavelengths is None:
                self._wavelengths = wls
            elif not np.all(np.abs(wls - self._wavelengths) < wl_tol * wls):
                raise ValueError('Mismatching wavelengths in dataset!')
            assert np.std(np.diff(self._wavelengths)) < 1e-3
            
            wl_I = splitext(f)[0].split('_')  # LED nominal wavelength, test current, vendor + purchase data
            wl_str = wl_I[0]
            wl_num = float(''.join([c for c in wl_str if c.isdigit()]))
            tag = ' '.join([str(math.floor(wl_num + 0.5)), wl_I[2]])
            if tag not in self.all_tags:
                led = LED()
                led.tag = tag
                led.nominal_wavelength = wl_num
                led.I_nominal = self.I_nominal
                led.spectrum = wls
                self._led_list.append(led)
            led_id = self.all_tags.index(tag)
            self._led_list[led_id].spectrum = np.vstack([self._led_list[led_id].spectrum, intensities])

        self._arrange_led_channel_list(initializing=True)
        return
    
    def _load_led_fit_pars(self, fname_fit_results):
        if fname_fit_results is None:
            return
        else:
            if not exists(fname_fit_results):
                return
        df_led_fit = pd.read_csv(fname_fit_results)
        nom_wls = df_led_fit['Nom. wavelength[nm]'].to_numpy(dtype=float)
        vendor_purchasedate = df_led_fit['vendor + purchase date']
        for i in range(len(nom_wls)):
            tag = ' '.join([str(math.floor(nom_wls[i] + 0.5)), vendor_purchasedate[i]])
            if tag not in self.all_tags:
                self._led_list.append(LED(
                    tag=tag,
                    nominal_wavelength=nom_wls[i],
                    I_nominal=self.I_nominal,
                ))
            id_led = self.all_tags.index(tag)
            fit_data_dict = {
                'etakBT_e':  df_led_fit['etakBT_e[V]'][i],
                'V_ref':     df_led_fit['V_ref[V]'][i],
                'V_loss':    df_led_fit['V_loss[V]'][i],
                'satuation': df_led_fit['satuation'][i],
            }
            self._led_list[id_led].fit_data = fit_data_dict
        return
    
    def _load_channel_data(self, pathname, channel_info_file:str='Summary_channels.csv'):
        df_channels = pd.read_csv(join(pathname, channel_info_file))
        self._channel_list = []  # clear channel list
        channel_ids = df_channels['Channel ID']
        N_channels = len(channel_ids)
        channel_components = df_channels['Component_name: count']
        for i_ch in range(N_channels):
            dict_comp = eval(('{' + channel_components[i_ch] + '}').replace('\t', '').strip(' ').replace(';', ','))
            assert type(dict_comp) is dict
            self._channel_list.append(dict_comp)

    def _arrange_led_channel_list(self, initializing=False):
        # ascendingly sort LEDs by nominal wavelengths
        led_order = np.argsort(self.all_nominal_wls)
        self._led_list = [self._led_list[x] for x in led_order]
        if initializing:
            # clean up secondary diffraction peaks
            wl_thresh_2nd_diffraction = 1.2 * np.max(self._wavelengths) / 2
            for i in range(self.N_leds):
                if self.led_nominal_wls[i] > wl_thresh_2nd_diffraction:
                    continue
                spec = self._led_list[i].spectrum
                nonzero_mask = (spec[0, :] - (self.led_nominal_wls[i] * 1.6) < 0)
                for j in range(1, spec.shape[0]):
                    self._led_list[i].spectrum[j] *= nonzero_mask
            # descendingly sort fluxes of each LED
            for i in range(self.N_leds):
                spec = self._led_list[i].spectrum
                fluxes = trapezoid(y=spec[1:, :], x=spec[0, :], axis=1)
                flux_order = np.flip(np.argsort(fluxes))
                self._led_list[i].spectrum[1:, :] = self._led_list[i].spectrum[1:, :][flux_order, :]

        # remove empty channels if any
        leds_exist_in_channels = np.zeros(self.N_channels)
        for i in range(self.N_channels):
            ch = self._channel_list[i]
            for k in ch.keys():
                led = self._led_list[self.all_tags.index(k)]
                if led.nominal_wavelength > 0 and ch[k] > 0:
                    leds_exist_in_channels[i] = 1
                    break
        self._channel_list = [ch for i, ch in enumerate(self._channel_list) if leds_exist_in_channels[i] > 0.5]

        # ascendingly sort channels by the nominal wavelength of each channel
        channel_order = np.argsort(self.get_channel_nom_wls())
        self._channel_list = [self._channel_list[x] for x in channel_order]
        return

    def _gen_leds_and_channels(self):
        '''Generate spectrum data'''
        peak_wls = np.array([335, 345,  # LEDs outside the range (360, 1000) are only used for calc. flux and not emitting any light
                             365, 385,  # integer (360 - 1000) = supplier 0 (default), .1 = supplier 1, .2 = supplier 2
                             405, 425, 445.2, 465, 485, 
                             505, 525, 530, 560.2, 575.1, 590.2, 
                             605, 625, 640.1, 660, 680,
                             700, 715.2, 740, 760.2, 780.2,
                             805, 830.2, 850, 880,  
                             905, 940, 980, 1010, 1020
                             ])  # peak wavelengths
        hwhm = np.array([8., 8.,
                         8., 8.,
                         8., 8., 10, 12, 14, 14,
                         15, 21, 21, 18, 14,
                         12, 10, 9., 9., 8.,
                         10, 10, 10, 10, 10,
                         17, 18, 16, 15,
                         18, 20, 22, 8., 8.
                         ])  # full-width at half maximum
        skew = np.array([0, 0, 
                         -0.1, -0.0, 
                         0.0, -0.2, -0.1, -0.2, 0.0, 0.1,
                         -0.1, -0.1, -0.1, 0, 0, 
                         0.1, 0.1, 0.2, 0.2, 0.2,
                         0.1, 0.0, 0.1, 0.0, -0.1,
                         0.1, 0.0, 0.2, 0.3,
                         0, -0.1, 0.1, 0, 0
                         ])
        
        np.random.seed(20240303)
        peak_shift = (2 * np.random.random(peak_wls.size) - 1) * hwhm / 2
        fluxes = np.array([1.0, 0.5, 0.2, 0.05])  # relative flux
        shift_ratios = np.array([0.0, 0.3, 0.7, 1.0])
        N_fluxes = fluxes.size
        N_leds = len(peak_wls)

        hwhm_norm_diff = np.sqrt(2 * np.log(2))  # exp(hwhm_norm_diff ** 2 / 2)
        for i in range(N_leds):
            stddev = hwhm[i] * hwhm_norm_diff
            led = LED(
                tag=str(math.floor(peak_wls[i])) + ' virtual gaussian with skew',
                nominal_wavelength=peak_wls[i],
                I_nominal=0.7,
            )
            led.fit_data = {
                'etakBT_e':  0.08,
                'V_ref':     2.5 * 500 / peak_wls[i],
                'V_loss':    0.45,
                'satuation': 0.1,
            }
            led.spectrum = 1 * self._wavelengths  # deepcopy
            for j in range(N_fluxes):
                bias = (self._wavelengths - (peak_wls[i] + peak_shift[i] * shift_ratios[j])) / stddev  # normalized bias from mean
                skew_bias = skew[i] * bias
                spec = np.exp(-bias ** 2 / 2 * (np.sqrt(1 + (skew_bias) ** 2) - skew_bias))
                spec = spec / trapezoid(y=spec, x=self._wavelengths) * 10 * fluxes[j]
                led.spectrum = np.vstack([led.spectrum, spec])
            self._led_list.append(led)
            self._channel_list.append({led.tag:1})
        return
    
    def add_led(self, peak_wavelength:float, id_led_ref:int=0, N_new:int=1,
                control_channel_id=0, new_control_channel:bool=False):
        '''Add N_new LED(s) with set peak wavelength (nm, nanometers) by shifting the spectrum of the reference LED, 
            whose index in self._led_list is id_led_ref.
        If new_control_channel is False, the new LED(s) is assigned to the control channel specified by control_channel_id
        If new_control_channel is True, the new LED(s) is assigned to a new channel and control_channel_id is ignored'''
        led_ref = self._led_list[id_led_ref]
        peak_wl_ref = led_ref.spectrum[0, np.argmax(led_ref.spectrum[1, :])]  # peak wavelength at max. flux
        wl_shift = peak_wavelength - peak_wl_ref  # from ref to new

        wavelengths_new = self._wavelengths
        f = interp1d(x=led_ref.spectrum[0, :], y=led_ref.spectrum[1:, :], axis=1, bounds_error=False, fill_value=0.)
        intensities_new = f(wavelengths_new - wl_shift)
        spectrum_new = np.vstack([wavelengths_new, intensities_new])

        led_new = LED(
            tag=str(math.floor(peak_wavelength + 0.5))+' virtual from '+led_ref.tag,
            nominal_wavelength=peak_wavelength,
            I_nominal=led_ref.I_nominal,
        )
        led_new.spectrum = spectrum_new
        led_new.fit_data = {k:led_ref.fit_data[k] for k in led_ref.fit_data.keys()}  # deepcopy
        self._led_list.append(led_new)

        ch = {led_new.tag: N_new}
        if new_control_channel:
            self._channel_list.append(ch)
        else:
            self._channel_list[control_channel_id] |= ch

        self._arrange_led_channel_list()

    # def remove_channel(self):
    #     pass  # TODO

    # def merge_channel(self):
    #     pass  # TODO
    
    def get_LED_list(self) -> list[LED]:
        """Get list of all LED instances excluding plain diodes.
        
        Returns
        -------
        list[LED]
            List of LED objects with nominal_wavelength > 0
        """
        return [led for led in self._led_list if led.nominal_wavelength > 0]

    def get_LED_current(self, id_LED: int, flux_ratio: float) -> float:
        """Calculate absolute current for LED at given flux ratio.
        
        Parameters
        ----------
        id_LED : int
            Index of LED in _led_list
        flux_ratio : float
            Desired flux ratio (0-1) relative to nominal
            
        Returns
        -------
        float
            Absolute current in Amperes
        """
        I_ratio = self._led_list[id_LED].get_current_from_flux(flux_ratio=flux_ratio)
        I_ch = self.I_nominal * I_ratio
        return I_ch

    def get_LED_spectrum(self, id_LED: int, flux_ratio: float = 1.) -> tuple[np.ndarray, np.ndarray]:
        """Get spectrum for specific LED at given flux ratio.
        
        Parameters
        ----------
        id_LED : int
            Index of LED in _led_list
        flux_ratio : float, optional
            Flux ratio (0-1), by default 1.0 (full power)
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (wavelengths, intensities) where:
            - wavelengths: Array of wavelength values in nm
            - intensities: Array of spectral intensities in W*m^-2*nm^-1
            
        Notes
        -----
        Returns zero spectrum if id_LED is out of range or LED is plain diode.
        """
        # if not (0 <= flux_ratio <= 1):
        #     raise ValueError('flux_ratio out of range! 0 <= flux_ratio <= 1, got flux_ratio = ' + str(flux_ratio))
        wl_mask = np.logical_and(self.min_wl <= self._wavelengths, self._wavelengths <= self.max_wl)
        wavelengths = self._wavelengths[wl_mask]
        flux_ratio = max(0., flux_ratio)
        if not (0 <= id_LED < self.N_diodes):  # if id_LED is out of range
            return wavelengths, wavelengths * 0.
        else:
            led = self._led_list[id_LED]
            if led.spectrum is None:  # plain diode
                led_spectrum = wavelengths * 0.
            else:
                led_spectrum = led.get_spectrum(flux_ratio=flux_ratio, wavelengths=wavelengths)
            return wavelengths, led_spectrum
    
    def get_channel_nom_wls(self) -> np.ndarray:
        """Get nominal wavelengths of first component in each channel.
        
        Returns
        -------
        np.ndarray
            Array of nominal wavelengths in nm
        """
        ch_wls = np.array([self.all_nominal_wls[
            self.all_tags.index(list(ch.keys())[0])
        ] for ch in self._channel_list])
        return ch_wls

    # def get_channel_current(self, id_channel: int, flux_ratio: Optional[float] = None) -> float:
    #     """Calculate absolute current for channel at given flux ratio.
        
    #     Parameters
    #     ----------
    #     id_channel : int
    #         Index of channel in _channel_list
    #     flux_ratio : Optional[float], optional
    #         Flux ratio (0-1), by default uses last calculated ratio
            
    #     Returns
    #     -------
    #     float
    #         Absolute current in Amperes
            
    #     Notes
    #     -----
    #     Currently only considers first LED in channel for calculation.
    #     """
    #     ch = self._channel_list[id_channel]
    #     id_leds_ch = [self.all_tags.index(k) for k in ch.keys()]
    #     if flux_ratio is None:
    #         flux_ratio_ch = self._channel_flux_ratios[id_channel]
    #     else:
    #         flux_ratio_ch = flux_ratio
    #     I_ch = self.get_LED_current(id_LED=id_leds_ch[0], flux_ratio=flux_ratio_ch)  # TODO based on only the first LED in channel
    #     return I_ch

    # def get_channel_voltage(self, id_channel: int, flux_ratio: Optional[float] = None) -> float:
    #     """Calculate required voltage for channel at given flux ratio.
        
    #     Parameters
    #     ----------
    #     id_channel : int
    #         Index of channel in _channel_list
    #     flux_ratio : Optional[float], optional
    #         Flux ratio (0-1), by default uses last calculated ratio
            
    #     Returns
    #     -------
    #     float
    #         Required voltage in Volts
            
    #     Notes
    #     -----
    #     Includes voltage drops from wiring resistance and all diodes in channel.
    #     """
    #     ch = self._channel_list[id_channel]
    #     if flux_ratio is None:
    #         flux_ratio_ch = self._channel_flux_ratios[id_channel]
    #     else:
    #         flux_ratio_ch = flux_ratio
    #     I_ch = self.get_channel_current(id_channel=id_channel, flux_ratio=flux_ratio_ch)
    #     V_ch = 0.
    #     for k in ch:
    #         led = self._led_list[self.all_tags.index(k)]
    #         if led.nominal_wavelength > 0:  # is an LED
    #             V_ch += led.get_voltage_from_flux(flux_ratio=flux_ratio_ch) * ch[k]
    #         else:  # is a plain diode
    #             V_ch += led.get_voltage_from_flux(flux_ratio=I_ch) * ch[k]
    #     if hasattr(self.R_wirings, '__len__'):
    #         R_w = self.R_wirings[id_channel]
    #     else:
    #         R_w = self.R_wirings
    #     V_ch += R_w * I_ch
    #     return V_ch

    def get_channel_spectrum(self, id_channel: int, flux_ratio: float = 1.) -> tuple[np.ndarray, np.ndarray]:
        """Get combined spectrum for channel at given flux ratio.
        
        Parameters
        ----------
        id_channel : int
            Index of channel in _channel_list
        flux_ratio : float, optional
            Flux ratio (0-1), by default 1.0 (full power)
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (wavelengths, intensities) where:
            - wavelengths: Array of wavelength values in nm
            - intensities: Array of spectral intensities in W*m^-2*nm^-1
            
        Notes
        -----
        Currently applies same flux ratio to all LEDs in channel.
        """
        # TODO ignored the fact that LEDs in the same channel must have the same current (therefore same I_ratio)
        #       instead of the same flux_ratio, unless the LEDs in this channel have the same satuation
        ch = self._channel_list[id_channel]
        channel_spectrum = np.sum([ch[k] * self.get_LED_spectrum(id_LED=self.all_tags.index(k), 
                                                                 flux_ratio=flux_ratio)[1]
                                   for k in ch.keys()], 
                                   axis=0)
        return self.get_LED_spectrum(id_LED=-1)[0], channel_spectrum
    
    def calc_channel_flux_ratios(self, wavelengths: np.ndarray, target_spectrum: np.ndarray, 
                                 tikhonov: float = 1e-2, max_iter: int = 10, tol: float = 1e-3, 
                                 blur_radius: Optional[float] = None) -> np.ndarray:
        """Calculate optimal channel flux ratios to match target spectrum.
        
        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelength sampling points for target spectrum (nm)
        target_spectrum : np.ndarray
            Desired spectral intensities (W*m^-2*nm^-1)
        tikhonov : float, optional
            Regularization factor to prevent unnecessary flux variations, by default 1e-2
        max_iter : int, optional
            Maximum iterations for optimization, by default 10
        tol : float, optional
            Convergence tolerance for flux ratios, by default 1e-3
        blur_radius : Optional[float], optional
            Gaussian blur radius (nm) to apply before optimization, by default None
            
        Returns
        -------
        np.ndarray
            Optimized flux ratios (0-1) for each channel
            
        Notes
        -----
        Uses quadratic programming to solve:
        min ||A*x - b||^2 + λ||M*x||^2
        where:
        - A is channel spectra matrix
        - b is target spectrum
        - M is second-order differential operator
        - λ is tikhonov regularization factor
        
        May raise ValueError if wavelengths are unevenly spaced when blur_radius is used.
        """
        wl_mask = np.logical_and(self.min_wl <= self._wavelengths, self._wavelengths <= self.max_wl)
        f = interp1d(x=wavelengths, y=target_spectrum)
        target_spectrum_vector = f(self._wavelengths[wl_mask])
        from quadprog import solve_qp as qp  # this module requires C compiler
        # matrix equation to solve: 
        # (spectrum_channels @ spectrum_channels.T) @ led_power = (spectrum_channels @ target_spectra)
        # A @ led_power = b
        M = np.array([[(i==j) - (i==j+1) for j in range(self.N_channels)] for i in range(self.N_channels)])
        M[0][0] -= 1  # M: second-order differential operator

        channel_flux_ratios = np.ones(self.N_channels)
        anneal_end = 1. / max_iter  # final (minimum) under-relaxation factor
        scaling_factor = 1.
        if blur_radius is not None:
            blur_kernel = np.exp(-((np.arange(-blur_radius * 3, blur_radius * 3 + 1)) / blur_radius) ** 2 / 2)  # gaussian kernel
            blur_kernel /= np.sum(blur_kernel)
            if np.std(np.diff(self._wavelengths[wl_mask])) > 1e-3:
                raise ValueError('Blurring is only applicable to arithmatically (evenly) spaced wavelengths, '+\
                                 'while self._wavelengths is not arithmatically (evenly) spaced.')
        for iter_count in range(max_iter):
            spectrum_channels = np.vstack([self.get_channel_spectrum(id_channel=i, flux_ratio=ch_flux_ratio)[1] / 
                                           (1e-9 + ch_flux_ratio) for i, ch_flux_ratio in enumerate(channel_flux_ratios)])
            if blur_radius is not None:
                spectrum_channels = np.vstack([np.convolve(spectrum_ch, blur_kernel, mode='same') for spectrum_ch in spectrum_channels])
                target_spectrum_vector = np.convolve(target_spectrum_vector, blur_kernel, mode='same')
            G = spectrum_channels @ spectrum_channels.T + tikhonov * (M @ M.T + 0.1 * np.eye(self.N_channels))
            a = spectrum_channels @ target_spectrum_vector / scaling_factor
            bounding_mask = np.eye(len(a))
            res = qp(G, a, bounding_mask, np.zeros(len(a)))
            channel_flux_ratios_new = res[0] * scaling_factor
            scaling_factor = np.average(channel_flux_ratios_new)
            errsq = np.sum((channel_flux_ratios_new - channel_flux_ratios)**2) / np.sum(channel_flux_ratios**2)
            # urf: under-relaxation factor, 0 < urf <= 1, cosine annealing
            urf = (0.5 * np.cos(iter_count * np.pi / (max_iter - 1)) + 0.5) * (1 - anneal_end) + anneal_end
            channel_flux_ratios = channel_flux_ratios_new * urf + channel_flux_ratios * (1 - urf)
            # print(urf, errsq)
            if errsq < tol**2:
                break
        channel_flux_ratios = np.maximum(0., channel_flux_ratios)
        self._channel_flux_ratios = channel_flux_ratios
        return channel_flux_ratios
    
    def output_spectrum(self, channel_flux_ratios: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """Get combined spectrum for all channels at given flux ratios.
        
        Parameters
        ----------
        channel_flux_ratios : Optional[np.ndarray], optional
            Flux ratios (0-1) for each channel, by default uses last calculated ratios
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (wavelengths, intensities) where:
            - wavelengths: Array of wavelength values in nm
            - intensities: Array of spectral intensities in W*m^-2*nm^-1
        """
        if channel_flux_ratios is None:
            channel_flux_ratios = self._channel_flux_ratios
        wl_mask = np.logical_and(self.min_wl <= self._wavelengths, self._wavelengths <= self.max_wl)
        spectrum_channels = np.vstack([self.get_channel_spectrum(id_channel=i, flux_ratio=channel_flux_ratios[i])[1] 
                                   for i in range(self.N_channels)])
        return self._wavelengths[wl_mask], np.sum(spectrum_channels, axis=0)[wl_mask]

def flux2current(flux_ratio: float, satuation: float) -> float:
    """Convert flux ratio to current ratio using saturation model.
    
    Parameters
    ----------
    flux_ratio : float
        Desired flux ratio (0-1)
    satuation : float
        Saturation parameter (>= 0)
        
    Returns
    -------
    float
        Current ratio (I/I_nominal)
        
    Notes
    -----
    Implements the equation:
    flux_ratio = (1 - exp(-satuation * I_ratio)) / (1 - exp(-satuation))
    """
    if flux_ratio > 1e-8 and satuation > 1e-8:
        flux_ratio = min(1.1, flux_ratio)
        I_ratio = np.log(1 - flux_ratio * (1 - np.exp(-satuation))) / (-satuation)
    else:
        if satuation > 1e-8:
            I_ratio = flux_ratio * satuation / (1. - np.exp(-satuation))
        else:
            I_ratio = flux_ratio * (1. + 0.5 * satuation)
    return I_ratio

def current2flux(I_ratio: float, satuation: float) -> float:
    """Convert current ratio to flux ratio using saturation model.
    
    Parameters
    ----------
    I_ratio : float
        Current ratio (I/I_nominal)
    satuation : float
        Saturation parameter (>= 0)
        
    Returns
    -------
    float
        Resulting flux ratio (0-1)
        
    Notes
    -----
    Implements the inverse of flux2current():
    - For satuation > 1e-8: uses full saturation model
    - For small satuation: uses linear approximation
    """
    if satuation > 1e-8:
        flux_ratio = (1 - np.exp(-satuation * I_ratio)) / (1 - np.exp(-satuation))
    else:
        flux_ratio = I_ratio / (1. + 0.5 * satuation)
    return flux_ratio

def flux2voltage(flux_ratio: float, etakBT_e: float, V_ref: float, 
                 V_loss: float, satuation: float) -> float:
    """Convert flux ratio to required voltage using diode model.
    
    Parameters
    ----------
    flux_ratio : float
        Desired flux ratio (0-1)
    etakBT_e : float
        Thermal voltage (eta * k_B * T / e)
    V_ref : float
        Reference voltage
    V_loss : float
        Series resistance voltage drop (I_nominal * R_s)
    satuation : float
        Current saturation parameter
        
    Returns
    -------
    float
        Required voltage in Volts
        
    Notes
    -----
    Implements diode equation:
    V = etakBT_e * ln(I_ratio + Is_Inom) + V_ref + I_ratio * V_loss
    where Is_Inom = exp(-V_ref / etakBT_e)
    
    See: https://lampz.tugraz.at/~hadley/psd/L6/pnIV.php
    """
    I_ratio = flux2current(flux_ratio=flux_ratio, satuation=satuation)
    Is_Inom = np.exp(-V_ref / etakBT_e)
    V = etakBT_e * np.log(I_ratio + Is_Inom) + V_ref + I_ratio * V_loss
    return V

def voltage2flux(voltage: float, etakBT_e: float, V_ref: float,
                 V_loss: float, satuation: float) -> float:
    """Convert applied voltage to resulting flux ratio.
    
    Parameters
    ----------
    voltage : float
        Applied voltage in Volts
    etakBT_e : float
        Thermal voltage (eta * k_B * T / e)
    V_ref : float
        Reference voltage
    V_loss : float
        Series resistance voltage drop (I_nominal * R_s)
    satuation : float
        Current saturation parameter
        
    Returns
    -------
    float
        Resulting flux ratio (0-1)
        
    Notes
    -----
    Numerically solves the diode equation from flux2voltage().
    May raise ValueError if voltage outside operating range.
    """
    def f(lnflux_ratio):
        return flux2voltage(np.exp(lnflux_ratio), etakBT_e, V_ref, V_loss, satuation) - voltage
    guess_left = np.log(1e-9)
    guess_right = np.log(3.)
    if f(guess_left) > 0:
        return np.exp(guess_left)
    if f(guess_right) < 0:
        return np.exp(guess_right)
    from scipy.optimize import root_scalar
    lnflux_ratio = root_scalar(f, bracket=(guess_left, guess_right), method='brentq').root
    return np.exp(lnflux_ratio)

def read_spectrum_file(fname: str, wavelengths: Optional[np.ndarray] = None,
                      fname_calibration: Optional[str] = None, 
                      intensity_column: int = 1) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Read spectral data from CSV file, optionally applying calibration.
    
    Parameters
    ----------
    fname : str
        Path to spectrum data file
    wavelengths : Optional[np.ndarray], optional
        Target wavelengths for resampling, by default None (return native sampling)
    fname_calibration : Optional[str], optional
        Path to calibration file, by default None
    intensity_column : int, optional
        Column index containing intensity data, by default 1
        
    Returns
    -------
    Union[tuple[np.ndarray, np.ndarray], np.ndarray]
        If wavelengths=None: returns (wavelengths, intensities)
        If wavelengths specified: returns intensities only
        
    Notes
    -----
    - Automatically skips header lines
    - Applies calibration if fname_calibration is provided and newer than data file
    - Uses linear interpolation for resampling
    """
    with open(fname, 'r', errors='ignore') as f:
        lines = f.readlines()
    start_line_num = 0
    for i, line in enumerate(lines):
        if line in ('wavelength\n', 'wavelength,\n') or 'wavelength \\' in line:
            start_line_num = i + 1
            break
    lines = lines[start_line_num:]
    lines = np.array([[float(entry) for entry in line.split(',')] for line in lines])
    wavelengths_file = lines[:, 0]
    intensities_file = lines[:, intensity_column]
    if fname_calibration is not None:
        intesity_calibration = read_spectrum_file(fname_calibration)[1]  # assertion: calibration was sampled on the same wavelength grid
        if getmtime(fname_calibration) > getmtime(fname):
            intensities_file *= intesity_calibration
    if wavelengths is None:
        return wavelengths_file, intensities_file
    else:  # wavelengths is specified
        func = interp1d(x=wavelengths_file, y=intensities_file, bounds_error=False, fill_value=0.)
        intensities = func(wavelengths)
        return intensities

def get_led_list_from_fit_results(fname_led_fit_results:str, I_nominal:float=0.7):
    h_light = HyperspectralLight(led_data_path=None)
    h_light.I_nominal = I_nominal
    h_light._led_list = []
    h_light._load_led_fit_pars(fname_fit_results=fname_led_fit_results)
    return h_light._led_list

def get_default_channel_list_from_fit_results(fname_fit_results='LEDtestdata/Summary_fit_results.csv'):
    df_fit_res = pd.read_csv(fname_fit_results)
    channel_list = []
    nom_wls = df_fit_res['Nom. wavelength[nm]'].to_numpy(dtype=float)
    led_mask = (nom_wls > 0)  # exclude the plain diode with nom_wl == -1
    nom_wls = nom_wls[led_mask]
    vendor_purchasedate = df_fit_res['vendor + purchase date'][led_mask]
    default_count = 1
    for i in range(len(nom_wls)):
        tag = ' '.join([str(math.floor(nom_wls[i] + 0.5)), vendor_purchasedate[i]])
        channel_list.append({tag: default_count})
    return channel_list


if __name__ == '__main__':
    h_light = HyperspectralLight()
    print(h_light.get_channel_voltage(id_channel=11, flux_ratio=0.5/h_light.I_nominal))
