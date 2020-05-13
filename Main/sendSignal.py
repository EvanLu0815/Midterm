import numpy as np

import serial

import time

Do = 261
Re = 294
Mi = 330
Fa = 349
So = 392
La = 440
So_ = 196

waitTime = 0.1


# generate the waveform table

signalLength = 42

t = np.linspace(0, 42, signalLength)

signalTable0 = [

  261, 261, 392, 392, 440, 440, 392,

  349, 349, 330, 330, 294, 294, 261,

  392, 392, 349, 349, 330, 330, 294,

  392, 392, 349, 349, 330, 330, 294,

  261, 261, 392, 392, 440, 440, 392,

  349, 349, 330, 330, 294, 294, 261]

signalTable1 = [

  So, Mi, Mi, Fa, Re, Re,

  Do, Re, Mi, Fa, So, So, So,

  So, Mi, Mi, Fa, Re, Re,

  Do, Mi, So, So, Mi,

  Re, Re, Re, Re, Re, Mi, Fa,

  Mi, Mi, Mi, Mi, Mi, Fa, So,

  So, Mi, Mi, Fa, Re, Re,

  Do, Mi, So, So, Do]

signalTable2 = [

  Do, Re, Mi, Do, Do, Re, Mi, Do,
  Mi, Fa, So, Mi, Fa, So,
  So, La, So, Fa, Mi, Do,
  So, La, So, Fa, Mi, Do,
  Do, So_, Do,
  Do, So_, Do]

noteLength0 = [

  1, 1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 1, 2]

noteLength1 = [

  1, 1, 2, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2, 
  1, 1, 2, 1, 1, 2, 
  1, 1, 1, 1, 4,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2, 
  1, 1, 2, 1, 1, 2,
  1, 1, 1, 1, 4]

noteLength2 = [

  1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 2, 1, 1, 2,
  1, 1, 1, 1, 2, 2,
  1, 1, 1, 1, 2, 2, 
  2, 2, 2, 2, 2, 2]

# output formatter

formatter = lambda x: "%3d" % x


# send the waveform table to K66F

serdev = '/dev/ttyACM0'

s = serial.Serial(serdev)

print("Sending signal ...")

print("It may take about %d seconds ..." % (int(signalLength * waitTime)))

line = int(s.readline())

# line = int(line)

print("line = %d" % line)

if line == 0:
    s.write(bytes(formatter(42), 'UTF-8'))
    for data in signalTable0:
        # print("%d\n" %data)

        s.write(bytes(formatter(data), 'UTF-8'))

        time.sleep(waitTime)
    for data in noteLength0:
        # print("%d\n" %data)

        s.write(bytes(formatter(data), 'UTF-8'))

        time.sleep(waitTime)
elif line == 1:
    s.write(bytes(formatter(49), 'UTF-8'))
    for data in signalTable1:

        # print("%d\n" %data)

        s.write(bytes(formatter(data), 'UTF-8'))

        time.sleep(waitTime)
    for data in noteLength1:
        # print("%d\n" %data)

        s.write(bytes(formatter(data), 'UTF-8'))

        time.sleep(waitTime)
elif line == 2:
    s.write(bytes(formatter(32), 'UTF-8'))
    for data in signalTable2:

        # print("%d\n" %data)

        s.write(bytes(formatter(data), 'UTF-8'))

        time.sleep(waitTime)
    for data in noteLength2:
        # print("%d\n" %data)

        s.write(bytes(formatter(data), 'UTF-8'))

        time.sleep(waitTime)
s.close()

print("Signal sended")