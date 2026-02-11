import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import datetime

guit_sample_rate, guit_signal = sp.io.wavfile.read('note_guitare_lad.wav')
basson_sample_rate, basson_signal = sp.io.wavfile.read('note_basson_plus_sinus_1000_hz.wav')

def amplitude_db(signal, eps=1e-12):
    return 20 * np.log10(np.maximum(np.abs(signal), eps))

def guitare():
    guit_fenetre = guit_signal * np.hanning(len(guit_signal))
    spectre = np.fft.fft(guit_fenetre)
    freqs = np.fft.fftfreq(len(guit_fenetre), 1 / guit_sample_rate)
    plt.figure()
    plt.plot(freqs, amplitude_db(spectre), label='Spectre')
    plt.title('Spectre de la guitare (dB)')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.legend()
    plt.grid()
    plt.show()
    
def enveloppe(signal, plot = False):
    filter_order = np.int32(filter_calcs(plot=plot))
    signal_abs = np.abs(signal)
    low_pass_filter = np.ones(filter_order) / (filter_order)
    enveloppe = np.convolve(signal_abs, low_pass_filter)
    
    if plot:  
        plt.figure()
        # plt.plot(signal_abs, label='|Signal|')
        plt.plot(enveloppe, label='Enveloppe')
        plt.title('Enveloppe temporelle')
        plt.xlabel('Échantillons')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()
    
    return enveloppe

def filter_calcs(plot=False):
    w = np.pi / 1000

    N = np.linspace(1, 1000, 1000) # ajuster pour trouver H(pi/1000) = -3db = 0.708
    # N = 884 donne H(pi/1000) = 0.708
    H_w = 1/N * (1 - np.exp(-1j * w * N))/(1 - np.exp(-1j * w))
    
    best_N = 1
    for i in range(len(N)):
        if abs(np.abs(H_w[i]) - 0.708) < abs((np.abs(H_w[best_N])) - 0.708):
            best_N = i
    
    if plot:       
        w = np.linspace(0, np.pi/100, 1000)
        w[0] = 1e-12  # éviter la division par zéro à w=0
        H_w = 1/N[best_N] * (1 - np.exp(-1j * w * N[best_N]))/(1 - np.exp(-1j * w))
        plt.figure()
        plt.plot(w, amplitude_db(H_w), label=f'|H(w)| pour N = {N[best_N]:.0f}')
        plt.title('Réponse en fréquence du filtre')
        plt.xlabel('Fréquence (rad/s)')
        plt.ylabel('Amplitude (dB)')
        plt.legend()
        plt.grid()

    return N[best_N]

def harmoniques_top32(signal, sample_rate, distance=400, plot=False):
    signal_fenetre = signal * np.hanning(len(signal))
    spectre = np.fft.rfft(signal_fenetre)
    freqs = np.fft.rfftfreq(len(signal_fenetre), 1 / sample_rate)
    freq_res = freqs[1] - freqs[0]
    amp = np.abs(spectre)
    phase = np.angle(spectre)
    
    max_amp = np.max(amp[:np.int32(2 * distance / freq_res)])  # Ignorer les fréquences très basses
    max_amp_idx = np.argmax(amp[:np.int32(2 * distance / freq_res)])
    
    amp[:max_amp_idx] = 0
    print(f"Max amplitude: {max_amp}")
    
    peaks = sp.signal.find_peaks(amp, distance=distance / freq_res)[0]
    print(f"{len(peaks)} sommets trouvés")
    
    top_idx = peaks[np.argsort(amp[peaks])][-32:]
    top_idx = top_idx[np.argsort(freqs[top_idx])]
    
    max_amp = np.max(amp)
    max_amp_idx = np.argmax(amp)
    
    print('Harmoniques (Hz, amplitude (dB), phase):')
    i = 1
    for idx in top_idx:
        print(f'{i} | {freqs[idx]:9.2f} |  {amplitude_db(amp[idx]):.6g} | {phase[idx]:.2f}')
        i += 1      
    
    harmoniques = [(freqs[idx], amp[idx], phase[idx]) for idx in top_idx]
    
    if plot:
        plt.figure()
        plt.plot(freqs, amplitude_db(np.abs(spectre)), label='Spectre (dB)')
        plt.scatter(freqs[top_idx], amplitude_db(amp[top_idx]), color='r', label='Harmoniques')
        plt.title('Spectre et harmoniques (dB)')
        plt.xlabel('Fréquence (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.legend()
        plt.grid()

    return harmoniques
    
def synthese(harmoniques,sample_rate=guit_sample_rate, plot=False, signal_enveloppe=guit_signal):
    enveloppe_signal = enveloppe(signal_enveloppe, plot=plot)
    duration = len(enveloppe_signal) / sample_rate
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(t)
    
    for freq, amp, phase in harmoniques:
        signal += amp * np.sin(2 * np.pi * freq * t + phase)
    
    signal_enveloppe =  enveloppe_signal * signal
    signal_enveloppe_normalise = normalise(signal_enveloppe)
    if plot:
        plt.figure()
        plt.subplot(2, 1, 2)
        plt.title('Signal synthétisé')
        plt.plot(t, signal_enveloppe, 'r', label='Signal synthétisé')
        plt.xlabel('Temps (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()
        plt.subplot(2, 1, 1)
        plt.plot(t[:len(signal_enveloppe)], signal_enveloppe, 'b', label='Signal original')
        # plt.xlabel('Temps (s)')
        plt.ylabel('Amplitude')
        plt.title('Signal original')
        plt.legend()
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

def synth_5ft_symphonie_beethoven():
    harmoniques_lad = harmoniques_top32(guit_signal, guit_sample_rate)

    harmoniques_sol = transpose(harmoniques_lad, freq_cible=392)
    harmoniques_red = transpose(harmoniques_lad, freq_cible=311.1)
    harmoniques_silence = [(0, 0, 0)]
    harmoniques_fa = transpose(harmoniques_lad, freq_cible=349.2)
    harmoniques_re = transpose(harmoniques_lad, freq_cible=293.7)
    
    signal_lad = synthese(harmoniques_lad, guit_sample_rate)
    signal_sol = synthese(harmoniques_sol, guit_sample_rate)
    signal_red = synthese(harmoniques_red, guit_sample_rate)
    signal_silence = synthese(harmoniques_silence, guit_sample_rate)  
    signal_fa = synthese(harmoniques_fa, guit_sample_rate)
    signal_re = synthese(harmoniques_re, guit_sample_rate)
    
    signals =[signal_sol, signal_sol, signal_sol, signal_red, signal_silence,
                             signal_fa, signal_fa, signal_fa, signal_re]
           
    nb_of_samples = .5 * guit_sample_rate
    
    synth_signal = np.concatenate([np.concatenate([s[:int(nb_of_samples)] for s in signals])])
    
    return synth_signal

def coupe_bande(sample_rate, ordre=6000, f_min=960, f_max=1040, plot=False):
    N = ordre % 2 == 0 and ordre or ordre + 1
    n = np.arange(-N//2, N//2 + 1) # Indices centrés sur 0
    
    K = 2 * N * ((f_max - f_min) / 2) / sample_rate + 1
    print(f"K : {K}")
    n[N//2] = 1  # Éviter la division par zéro à n=0
    h_pb  = (1/N) * (np.sin(np.pi * n * K / N) / np.sin(np.pi * n / N))
    h_pb[N//2] = K/N
    h_pb = h_pb * np.blackman(N +1)
    
    dirac = np.zeros(N + 1)
    dirac[N//2] = 1
    h_cb = dirac - 2 * h_pb * np.cos(2 * np.pi * n * ((f_max + f_min) / 2) / sample_rate)
    
    if plot:
        H_cp = np.fft.fft(h_cb)
        H_pb = np.fft.fft(h_pb)
        
        H_cp_db = amplitude_db(H_cp)
        H_pb_db = amplitude_db(H_pb)
         
        freqs = np.fft.fftfreq(N, 1 / sample_rate)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title('Réponse en fréquence du coupe-bande')
        plt.plot(freqs[:N//4], H_cp_db[:N//4], label='Coupe-bande')
        plt.xlabel('Fréquence (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.xlim(0, 2000)
        plt.ylim(-60, 5)
        plt.legend()
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.title('Phase du coupe-bande')
        plt.plot(freqs[:N//4], np.angle(H_cp[:N//4]), label='Coupe-bande')
        plt.xlabel('Fréquence (Hz)')
        plt.ylabel('Phase (radians)')
        plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['-π', '-π/2', '0', 'π/2', 'π'])
        plt.xlim(0, 2000)

        plt.figure()
        plt.title('Réponse temporelle du coupe-bande')
        plt.plot(n, h_cb, label='Coupe-bande')
        plt.xlabel('Échantillons')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()
        
        plt.figure()
        plt.title('Réponse temporelle du coupe-bande à 1000 Hz')
        plt.plot(n, np.convolve(h_cb, np.cos(2 * np.pi * n * 1000 / sample_rate))[:len(n)], label='Coupe-bande à 1000 Hz')
        plt.xlabel('Échantillons')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()

    return h_cb

def filtre_basson(plot=False):
    filtre = coupe_bande(basson_sample_rate, ordre=6000, f_min=960, f_max=1040, plot=plot)
    signal_filtre = np.convolve(basson_signal, filtre)     

    if plot:
        spectre = np.fft.rfft(basson_signal * np.hanning(len(basson_signal)))
        freqs = np.fft.rfftfreq(len(basson_signal), 1 / basson_sample_rate)
        
        spectre_filtre = np.fft.rfft(signal_filtre * np.hanning(len(signal_filtre)))
        freqs_filtre = np.fft.rfftfreq(len(signal_filtre), 1 / basson_sample_rate)
    
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(freqs, amplitude_db(np.abs(spectre)), label='Signal original (dB)')
        plt.title('Spectres du signal original')
        plt.xlabel('Fréquence (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.grid()
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(freqs_filtre, amplitude_db(np.abs(spectre_filtre)), 'r', label='Signal filtré (dB)')
        plt.title('Spectres du signal filtré')
        plt.xlabel('Fréquence (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.grid()
        plt.legend()
        
    return signal_filtre

def normalise(signal):
    return signal / (np.max(np.abs(signal)) > 0 and np.max(np.abs(signal)) or 1)

def plot_basson_filtre_vs_synthese():    
    basson_filtre = filtre_basson(plot=True)    
    harmoniques_bass = harmoniques_top32(basson_filtre, basson_sample_rate, distance=200, plot=True)
    synth_basson = synthese(harmoniques_bass, basson_sample_rate, signal_enveloppe=basson_filtre, plot=True)    

    spectre_filtre = np.fft.rfft(basson_filtre * np.hanning(len(basson_filtre)))
    freqs_filtre = np.fft.rfftfreq(len(basson_filtre), 1 / basson_sample_rate)
    spectre_synth_basson = np.fft.rfft(synth_basson * np.hanning(len(synth_basson)))
    freqs_synth_basson = np.fft.rfftfreq(len(synth_basson), 1 / basson_sample_rate)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(freqs_filtre, amplitude_db(np.abs(spectre_filtre)), label='Signal filtré (dB)')
    plt.title('Spectre du signal filtré (dB)')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(freqs_synth_basson, amplitude_db(np.abs(spectre_synth_basson)), 'r', label='Signal synthétisé (dB)')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Spectre du signal synthétisé (dB)')
    plt.legend()
    plt.grid()

def plot_basson_original_vs_synthese():
    
    basson_filtre = filtre_basson(plot=True)    
    harmoniques_bass = harmoniques_top32(basson_filtre, basson_sample_rate, distance=200, plot=True)
    synth_basson = synthese(harmoniques_bass, basson_sample_rate, signal_enveloppe=basson_filtre, plot=True)
    
    basson_spectre = np.fft.rfft(basson_filtre * np.hanning(len(basson_filtre)))
    basson_freqs = np.fft.rfftfreq(len(basson_filtre), 1 / basson_sample_rate)
    
    synth_basson_spectre = np.fft.rfft(synth_basson * np.hanning(len(synth_basson)))
    synth_basson_freqs = np.fft.rfftfreq(len(synth_basson), 1 / basson_sample_rate)
    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Spectre du signal original (dB)')
    plt.plot(basson_freqs, amplitude_db(np.abs(basson_spectre)), label='Signal original (dB)')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title('Spectre du signal synthétisé (dB)')
    plt.plot(synth_basson_freqs, amplitude_db(np.abs(synth_basson_spectre)), 'r', label='Signal synthétisé (dB)')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.legend()
    
def plot_guitare_original_vs_synthese():
    harmoniques = harmoniques_top32(guit_signal, guit_sample_rate, plot=True)
    synth_signal = synthese(harmoniques, guit_sample_rate, signal_enveloppe=guit_signal, plot=True)
    
    spectre_guit = np.fft.rfft(guit_signal * np.hanning(len(guit_signal)))
    freqs_guit = np.fft.rfftfreq(len(guit_signal), 1 / guit_sample_rate)
    spectre_synth = np.fft.rfft(synth_signal * np.hanning(len(synth_signal)))
    freqs_synth = np.fft.rfftfreq(len(synth_signal), 1 / guit_sample_rate)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(freqs_guit, amplitude_db(np.abs(spectre_guit)), label='Signal original (dB)')
    plt.title('Spectre du signal original (dB)')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(freqs_synth, amplitude_db(np.abs(spectre_synth)), 'r', label='Signal synthétisé (dB)')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Spectre du signal synthétisé (dB)')
    plt.legend()

if __name__ == "__main__":
    plot = True
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    signal_to_wav(synthese(harmoniques_top32(guit_signal, guit_sample_rate, plot=plot), guit_sample_rate, plot=plot), filename=f"synthese/synthese_guitare_lad_{now}.wav")
    signal_to_wav(synth_5ft_symphonie_beethoven(), filename=f"synthese/symphonie_bethoven_{now}.wav")

    # Filtrage du basson
    signal_to_wav(normalise(filtre_basson(plot=False)), filename=f"synthese/basson_filtre_{now}.wav")
    
    # Synthèse du basson
    basson_filtre = filtre_basson(plot=plot)    
    harmoniques_bass = harmoniques_top32(basson_filtre, basson_sample_rate, distance=200, plot=plot)
    synth_basson = synthese(harmoniques_bass, basson_sample_rate, signal_enveloppe=basson_filtre, plot=plot)    
    signal_to_wav(synth_basson, basson_sample_rate, filename=f"synthese/synthese_basson_lad_{now}.wav")
    
    plot_guitare_original_vs_synthese()
    plot_basson_original_vs_synthese()
    
    plt.show()