from elice_utils import EliceUtils
import numpy as np
import matplotlib.pyplot as plt



elice_utils = EliceUtils()

def plt_show():
    plt.savefig("fig")
    elice_utils.send_image("fig.png")

def plot_signals(signals, titles):
    for i, signal in enumerate(signals):
        plt.figure(i)
        plt.plot(signal, label=titles[i])
        plt.legend()
        plt_show()

def convolution_1D_time_domain(signal, kernel):
    # Lengths of input signals
    signal=np.asarray(signal)
    n=len(signal)
    kernel=np.asarray(np.flip(kernel,axis=0))
    m=len(kernel)
    # Initialize the output signal with zeros
    output = np.zeros(n+m-1)

    # Slide the kernel over the signal
    for i in range(len(output)):
        for j in range(m):
            if i+j-m+1<0 or i+j-m+1>n-1:continue
            else: output[i] += kernel[j]*signal[i-(m-1-j)]
    # output=np.convolve(signal,kernel,mode='full')
    return output


def convolution_1D_freq_domain(signal, kernel):
    signal=np.asarray(signal)
    n = len(signal)
    kernel=np.asarray(kernel)
    m = len(kernel)
    padded_length = n+m-1
    # Pad signals
    padded_signal=np.pad(signal,(0,padded_length-n),'constant')
    padded_kernel=np.pad(kernel,(0,padded_length-m),'constant')
    # Take FFT of both padded signals
    signal_freq = np.fft.rfft(padded_signal)
    kernel_freq = np.fft.rfft(padded_kernel)
    # Multiply the frequency representations
    convoluted_freq = np.multiply(signal_freq,kernel_freq)
    # Take inverse FFT and only keep the real part
    result = np.fft.irfft(convoluted_freq)
    print(result.shape)
    return result

def main():
    signal = [2, 1, 2, 1]
    kernel = [1, 0, -1]
    result_time = convolution_1D_time_domain(signal, kernel)
    result_freq = convolution_1D_freq_domain(signal, kernel)
    print(result_time)
    print(result_freq)
    plot_signals([signal, kernel, result_time, result_freq], ["Signal", "Kernel", "Time Domain Convolution",                    "Frequency Domain Convolution"])



if __name__ == "__main__":
    main()
