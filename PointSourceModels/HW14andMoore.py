import numpy as np
#import pylab as plt
import matplotlib.pyplot as plt
from astropy import units as u

from astropy.io import fits
from astropy.table import Table

hw14file = 'hurley-walker14.fit'
hw14 = Table.read(hw14file)

# Are there any likely sources in H-W14 that could be the anomaloy in Moore 16?
candidates = (hw14['S180']>10)*(hw14['RAJ2000'] > 90)*(hw14['RAJ2000']<120)*(hw14['DEJ2000']<-30+45./2.)*(hw14['DEJ2000']>-30-45./2.)

wh_candidates = np.where(candidates)[0]

for wh in wh_candidates:
    print "%s %5.2f %5.2f %4.1f Jy" % (hw14['MWACS'][wh],hw14['RAJ2000'][wh]/15.,hw14['DEJ2000'][wh],hw14['S180'][wh])
