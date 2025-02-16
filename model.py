import crepe
from scipy.io import wavfile

sr, audio = wavfile.read("test.wav")
time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)

print(frequency.shape)
