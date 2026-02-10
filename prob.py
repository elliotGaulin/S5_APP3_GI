import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import datetime

guit_sample_rate, guit_signal = sp.io.wavfile.read('note_guitare_lad.wav')

def guitare():
    spectre = np.fft.fft(guit_signal)
    freqs = np.fft.fftfreq(len(guit_signal), 1 / guit_sample_rate)
    plt.figure()
    plt.plot(freqs, abs(spectre))
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()
    

def enveloppe(plot = False):
    filter_order = np.int32(filter_calcs())
    guit_signal_abs = np.abs(guit_signal)
    low_pass_filter = np.ones(filter_order) / (filter_order)
    guit_enveloppe = np.convolve(guit_signal_abs, low_pass_filter)
    
    if plot:
        plt.figure()
        plt.plot(guit_signal_abs)
        plt.plot(guit_enveloppe, 'r')
        plt.grid()
    
    return guit_enveloppe

def filter_calcs(plot=False):
    w = np.pi / 1000

    N = np.linspace(1, 1000, 1000) # ajuster pour trouver H(pi/1000) = -3db = 0.708
    # N = 884 donne H(pi/1000) = 0.708
    H_w = 1/N * (1 - np.exp(-1j * w * N))/(1 - np.exp(-1j * w))
    
    best_N = 1
    for i in range(len(N)):
        if abs(np.abs(H_w[i]) - 0.708) < abs((np.abs(H_w[best_N])) - 0.708):
            best_N = i
    
    # print(f"N optimal pour H(pi/1000) = -3db : {N[best_N]} avec H(pi/1000) = {np.abs(H_w[best_N])}")
    
    # print(f"N - 1 : {N[best_N] - 1} avec H(pi/1000) = {np.abs(H_w[best_N - 1])}")
    # print(f"N + 1 : {N[best_N] + 1} avec H(pi/1000) = {np.abs(H_w[best_N + 1])}")
    
    if plot:
        plt.figure()
        plt.scatter(N, np.abs(H_w))
        plt.grid()

    return N[best_N]

def harmoniques_top32_guitare(plot=False):
    spectrum = np.fft.rfft(guit_signal)
    freqs = np.fft.rfftfreq(len(guit_signal), 1 / guit_sample_rate)
    amp = np.abs(spectrum)
    phase = np.angle(spectrum)
    
    max_amp = np.max(amp)
    max_amp_idx = np.argmax(amp)
    
    amp[:max_amp_idx] = 0
    print(f"Max amplitude: {max_amp}")
    
    freq_res = freqs[1] - freqs[0]
    peaks = sp.signal.find_peaks(amp, distance=400 / freq_res)[0]
    print(f"{len(peaks)} sommets trouvés")
    
    top_idx = peaks[np.argsort(amp[peaks])][-32:]
    top_idx = top_idx[np.argsort(freqs[top_idx])]
    
    max_amp = np.max(amp)
    max_amp_idx = np.argmax(amp)
    
    print('Harmoniques (Hz, amplitude):')
    i = 1
    for idx in top_idx:
        print(f'#{i} | {freqs[idx]:9.2f} Hz  |  {amp[idx]:.6g}')
        i += 1
        
        
    
    harmoniques = [(freqs[idx], amp[idx], phase[idx]) for idx in top_idx]
    
    if plot:
        plt.figure()
        plt.plot(freqs, amp)
        plt.scatter(freqs[top_idx], amp[top_idx], color='r')
        plt.xlabel('Fréquence (Hz)')
        plt.ylabel('Amplitude')
        plt.grid()

    return harmoniques
    
def synthese(harmoniques,sample_rate=guit_sample_rate, plot=False):
    enveloppe_signal = enveloppe()
    duration = len(enveloppe_signal) / sample_rate
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(t)
    
    for freq, amp, phase in harmoniques:
        signal += amp * np.sin(2 * np.pi * freq * t + phase)
    
    signal_enveloppe =  enveloppe_signal * signal
    signal_enveloppe_normalise = signal_enveloppe / np.max(np.abs(signal_enveloppe))
    if plot:
        plt.subplot(2, 1, 1)
        plt.title('Signal synthétisé')
        plt.plot(t, signal_enveloppe, 'r', label='Signal synthétisé')
        plt.xlabel('Temps (s)')
        plt.ylabel('Amplitude')
        plt.subplot(2, 1, 2)
        plt.plot(t[:len(guit_signal)], guit_signal, 'b', label='Signal original')
        plt.xlabel('Temps (s)')
        plt.ylabel('Amplitude')
        plt.title('Signal original')
        plt.grid()
    return signal_enveloppe_normalise

def signal_to_wav(signal, sample_rate=guit_sample_rate, filename='synthese.wav'):
    signal_int16 = np.int16(signal * 32767)
    sp.io.wavfile.write(filename, sample_rate, signal_int16)

def transpose(harmoniques, freq_cible=440):
    freq_fondamentale = harmoniques[0][0]
    harmoniques_transposees = []
    for freq, amp, phase in harmoniques:
        freq_transposee = freq * (freq_cible / freq_fondamentale)
        harmoniques_transposees.append((freq_transposee, amp, phase))
    return harmoniques_transposees


def synth_5ft_symphonie_bethoven():
    harmoniques_lad = harmoniques_top32_guitare()

    harmoniques_sol = transpose(harmoniques_lad, freq_cible=392)
    harmoniques_red = transpose(harmoniques_lad, freq_cible=311.1)
    harmoniques_silence = [(0, 0, 0)]
    harmoniques_fa = transpose(harmoniques_lad, freq_cible=349.2)
    harmoniques_re = transpose(harmoniques_lad, freq_cible=293.7)
    
    signal_sol = synthese(harmoniques_sol)
    signal_red = synthese(harmoniques_red)
    signal_silence = synthese(harmoniques_silence)  
    signal_fa = synthese(harmoniques_fa)
    signal_re = synthese(harmoniques_re)
    
    signals =[signal_sol, signal_sol, signal_sol, signal_red, signal_silence,
                             signal_fa, signal_fa, signal_fa, signal_re]
        
    nb_of_samples = .5 * guit_sample_rate
    
    synth_signal = np.concatenate([np.concatenate([s[:int(nb_of_samples)] for s in signals])])
    
    return synth_signal

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    signal_to_wav(synthese(harmoniques_top32_guitare()), filename=f"synthese/synthese_guitare_lad_{now}.wav")
    signal_to_wav(synth_5ft_symphonie_bethoven(), filename=f"synthese/symphonie_bethoven_{now}.wav")

    # print(filter_calcs())
    plt.show()
    print("HERE")