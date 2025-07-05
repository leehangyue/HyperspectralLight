import pandas as pd
import numpy as np
from os import listdir
from os.path import split, join, isdir, splitext
from time import strptime
from calendar import timegm
from tqdm import tqdm

def merge_dir(dirname:str):
    fnames = listdir(dirname)
    fnames.sort()
    fnames = [fname for fname in fnames if fname.lower().endswith('.csv') and not fname.startswith('.')]
    titles = None
    timestrs = [None, ] * len(fnames)
    bulk_data = [None, ] * len(fnames)
    for i, fname in enumerate(tqdm(fnames)):
        timestrs[i] = splitext(fname)[0].split('_')[-1]
        full_fname = join(dirname, fname)
        df = pd.read_csv(full_fname, header=None)
        if titles is None:
            titles = df.iloc[:, 0]
        bulk_data[i] = df.iloc[:, 1]
    titles = np.insert(titles, 0, "Time")
    bulk_data = np.hstack([np.array(timestrs)[:, np.newaxis], bulk_data])
    merged_df = pd.DataFrame(np.vstack([titles, bulk_data]).T)
    merged_df.to_csv(join(dirname, 'merged.csv'), index=False, header=False)
    print('done.')

def merge_dir_spectra_only(dirname:str, interval=None, normalize=False, sum_flux=False):
    fnames = listdir(dirname)
    fnames.sort()
    fnames = [fname for fname in fnames if fname.lower().endswith('.csv') and not fname.startswith('.')]
    wavelengths = None
    timestrs = [None, ] * len(fnames)
    epoch_times = [None, ] * len(fnames)
    bulk_data = [None, ] * len(fnames)
    print("Reading files...")
    for i, fname in enumerate(tqdm(fnames)):
        timestr = splitext(fname)[0].split('_')[-1]
        try:
            timestamp = strptime(timestr, "%Y%m%d%H%M%S%f")
            epoch_time = timegm(timestamp) + float(timestr[-3:]) * 1e-3
        except ValueError:
            continue
        timestrs[i] = timestr
        epoch_times[i] = epoch_time
        full_fname = join(dirname, fname)
        df = pd.read_csv(full_fname, header=None)
        if wavelengths is None:
            wavelengths = df.iloc[51:, 0]
        bulk_data[i] = df.iloc[51:, 1]
    
    timestrs = [x for x in timestrs if x is not None]
    epoch_times = [x for x in epoch_times if x is not None]
    bulk_data = [x for x in bulk_data if x is not None]
    elapsed_times = np.array(epoch_times) - epoch_times[0]
    if sum_flux:
        from scipy.integrate import trapezoid
        normalize = False
        fname_out = 'merged_flux_only.csv'
        wl = np.array(wavelengths, dtype=np.float32)
        bulk_data = [[trapezoid(y=np.array(y, dtype=np.float32), x=wl), ] for y in bulk_data]
        wavelengths = ["Flux 340-1020 nm [W/m2]"]
    else:
        fname_out = 'merged_spectra_only.csv'
    if interval is None:
        bulk_data = np.hstack([elapsed_times[:, np.newaxis], np.array(timestrs)[:, np.newaxis], bulk_data])
        wavelengths = np.insert(wavelengths, 0, [f"wavelength \ Elapsed time since {timestrs[0]} [s]", "Abs. time"])
    else:
        assert isinstance(interval, int | float)
        assert interval > 0
        t_max = np.max(elapsed_times)
        t_new = np.arange(0., t_max, interval)
        t_new = np.append(t_new, t_new[-1] + interval)
        bulk_data = np.array(bulk_data, dtype=np.float32)
        bulk_data_new = [None, ] * len(t_new)
        fw = lambda dt: np.exp(-((dt) / interval) ** 2)
        print("Interpolating...")
        for i, t in enumerate(tqdm(t_new)):
            weights = fw(elapsed_times - t)
            weights /= np.sum(weights)
            bulk_data_new[i] = weights @ bulk_data
            if normalize:
                bulk_data_new[i] /= max(1e-6, np.max(bulk_data_new[i]))
        if normalize:
            fname_out = 'merged_spectra_only_normalized.csv'
        bulk_data = np.hstack([t_new[:, np.newaxis], bulk_data_new])
        wavelengths = np.insert(wavelengths, 0, [f"Elapsed time since {timestrs[0]} [s]", ])
    if sum_flux:
        merged_df = pd.DataFrame(np.vstack([wavelengths,  bulk_data]))
    else:
        merged_df = pd.DataFrame(np.vstack([wavelengths,  bulk_data]).T)
    merged_df.to_csv(join(dirname, fname_out), index=False, header=False)
    print('done.')

if __name__ == '__main__':
    dirname = input('Please specify the data folder:\n').strip('& ').strip('\"').strip('\'')
    if not isdir(dirname):
        dirname = split(dirname)[0]
    # test_data = merge_dir(dirname=dirname)
    test_data = merge_dir_spectra_only(dirname=dirname, interval=None, sum_flux=True)
    test_data = merge_dir_spectra_only(dirname=dirname, interval=15, sum_flux=False)
