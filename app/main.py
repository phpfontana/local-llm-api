import numpy as np
import pyaudio as pa
import struct
import sys
import matplotlib.pyplot as plt

CHUNK = 512
FORMAT = pa.paInt16
CHANNELS = 1 if sys.platform == 'darwin' else 2
RATE = 44100
RECORD_SECONDS = 5


p = pa.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

fig,ax = plt.subplots()
x = np.arange(0,2*CHUNK,2)
line, = ax.plot(x, np.random.rand(CHUNK),'r')
ax.set_ylim(-32770,32770)
ax.ser_xlim = (0,CHUNK)
fig.show()

while 1:
    data = stream.read(CHUNK, exception_on_overflow=False)
    dataInt = struct.unpack(str(CHUNK) + 'h', data)
    line.set_ydata(dataInt)
    fig.canvas.draw()
    fig.canvas.flush_events()