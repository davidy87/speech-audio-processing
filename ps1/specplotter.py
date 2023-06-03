import argparse
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.misc
import os.path

class SpecPlotter(object):
    def __init__(self):
        # signal processing stuff
        self.fnotch = 60
        self.notchQ = 30
        self.assumed_rate = 16000
        self.coeff = 0.97
        self.window_size = 0.004
        self.window_stride =0.001
        self.hop_length = int(self.assumed_rate * self.window_stride)
        self.n_fft = 1024
        self.win_length = int(self.assumed_rate * self.window_size)
        self.window = scipy.signal.hamming
        self.db_spread = 40
        self.db_cutoff = 3
        self.fig_height=10
        self.inches_per_sec = 10

    def compute_spectrogram(self, signal):
        g = scipy.signal.gaussian(41, std=6)
        g = g / g.sum()
        y = signal - signal.mean()
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.win_length,
                                                 hop_length=self.hop_length)
        zcr = np.convolve(zcr[0], g, mode='same')
        zcr = zcr - zcr.min()
        b, a = scipy.signal.iirnotch(self.fnotch, self.notchQ, self.assumed_rate)
        y = scipy.signal.lfilter(b, a, y)
        y = np.append(y[0],y[1:]-self.coeff*y[:-1])
#        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.win_length,
#                                                 hop_length=self.hop_length)
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.win_length, window=self.window)
        power_spec = np.abs(stft)**2
        total_energy = 10*np.log10(np.sum(power_spec, axis=0))
        total_energy = total_energy - total_energy.max()
        total_energy = np.clip(total_energy, -1 * self.db_spread, 0)
        total_energy = total_energy - total_energy.min()
        total_energy = np.convolve(total_energy, g, mode='same')
        total_energy = total_energy - total_energy.max()
        f0 = int(np.round((125 / self.assumed_rate * .5) * power_spec.shape[0]))
        f1 = int(np.round((750 / self.assumed_rate * .5) * power_spec.shape[0]))
        lowfreq_energy = 10*np.log10(np.sum(power_spec[f0:f1,:], axis=0))
        lowfreq_energy = lowfreq_energy - lowfreq_energy.max()
        lowfreq_energy = np.clip(lowfreq_energy, -1 * self.db_spread, 0)
        lowfreq_energy = lowfreq_energy - lowfreq_energy.min()
        lowfreq_energy = np.convolve(lowfreq_energy, g, mode='same')
        lowfreq_energy = lowfreq_energy - lowfreq_energy.max()
        #mel_basis = librosa.filters.mel(self.assumed_rate, self.n_fft, n_mels=80,
        #                                fmin=20)
        #power_spec = np.dot(mel_basis, power_spec)
        logspec = librosa.power_to_db(power_spec, ref=np.max)
        logspec = np.flipud(logspec)
        clipped_logspec = np.clip(logspec, -1*self.db_spread, -1*self.db_cutoff)
        
        return y, clipped_logspec, zcr, total_energy, lowfreq_energy
    
    def plot_spectrogram(self, signal, outfile=None):
        y, spec, zcr, te, lfe  = self.compute_spectrogram(signal)
        extent= [0, signal.shape[0] / self.assumed_rate, 0,
                self.assumed_rate / 2000] # Convert x to seconds, y to kHz
        n_sec = signal.shape[0] / self.assumed_rate
        figure = plt.figure(figsize=(n_sec * self.inches_per_sec, self.fig_height))
        gs = figure.add_gridspec(nrows=5, ncols=1, height_ratios=[1,1,1,12,1], hspace=0.05)
        ax = figure.add_subplot(gs[0,0])
        # plot zcr
        plt.fill_between(np.arange(len(zcr)), zcr, y2=zcr.min(), color='gray')
        plt.margins(0, 0)
        ax.set_ylim(0, 1)
        plt.xticks([], [])
        plt.annotate('Zero Crossing Rate', (10, 0.6))
        plt.ylabel('kHz')
        plt.yticks([], [])
        ax = figure.add_subplot(gs[1,0])
        # plot total energy
        plt.fill_between(np.arange(len(te)), te, y2=te.min(), color='gray')
        plt.margins(0, 0)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.ylabel('dB')
        plt.annotate('Total Energy', (10, (-15/40)*np.abs(te.min())))
        ax = figure.add_subplot(gs[2,0])
        # plot low freq energy
        plt.fill_between(np.arange(len(lfe)), lfe, y2=lfe.min(), color='gray')
        plt.margins(0, 0)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.ylabel('dB')
        plt.annotate('Energy: 125 to 750 Hz', (10, -(15/40)*np.abs(lfe.min())))
        ax = figure.add_subplot(gs[3,0])
        # plot spec
        plt.imshow(spec, cmap='gist_gray', extent=extent, aspect='auto')
        #plt.xlabel('Time (s)')
        plt.ylabel('Frequency (kHz)')
        plt.xticks(np.arange(0, n_sec, 0.1))
        ax.tick_params(labelbottom=False, labelleft=True, labelright=True)
        #ax = plt.gca()
        ax.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.annotate('Wide Band Spectrogram', (0.01, 7.7))
        ax = figure.add_subplot(gs[4,0])
        # plot waveform
        plt.plot(y, linewidth=0.25, color='k')
        plt.margins(0, 0)
        ticks = np.arange(0, n_sec, 0.1)
        ticklabs = ["%.1f" % z for z in ticks]
        plt.xticks(ticks=self.assumed_rate*ticks, labels=ticklabs)
        plt.yticks([], [])
        plt.xlabel('Time (seconds)')
        plt.annotate('Waveform', (200, 0.3 * y.max()))
        if outfile:
            plt.savefig(outfile, bbox_inches='tight')
        else:
            plt.show()
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('wav', type=str, help='Input .wav file')
    parser.add_argument('-s', type=float, default=0.0, help='Start second')
    parser.add_argument('-e', type=float, default=-1.0, help='End second')
    parser.add_argument('-o', type=str, default=None, help='Save .pdf to this file instead of plotting.')
    args = parser.parse_args()
    spec_plotter = SpecPlotter()
    
    file = os.path.basename("/Users/david/Desktop/speech_audio_proc/ps1/signal.wav")
    x, sr = librosa.load(file, sr=spec_plotter.assumed_rate)

    start_sample = int(args.s * spec_plotter.assumed_rate)
    if args.e < 0:
        end_sample = x.shape[0]
    else:
        end_sample = int(args.e * spec_plotter.assumed_rate)
    x = x[start_sample:end_sample]
    spec_plotter.plot_spectrogram(x, outfile=args.o)
