"""
functions to read/write data to specialized pickle files and various formats
"""

import cPickle as pkl
import numpy as np

def writeImgPkl(fn,d,res=None,sb=None,rcumode=None,station=None,int_time=None,fttype=None,imtype=None):
    """Write an image cube to a pickle file"""
    imgDict={
        'res':res,
        'sb':sb,
        'rcumode':rcumode,
        'station':station,
        'int_time':int_time,
        'fttype':fttype,
        'imtype':imtype,
        'img':d
    }
    fh=open(fn,'wb')
    pkl.dump(imgDict,fh)
    fh.close()

def readImgPkl(fn):
    """Read an image cube from a pickle file"""
    fh=open(fn,'rb')
    imgDict=pkl.load(fh)
    fh.close()
    return imgDict

def writeFEKOPkl(fn,feko):
    """Write a FEKO instance to a pickle file"""
    fh=open(fn,'wb')
    pkl.dump(feko,fh)
    fh.close()

def readFEKOPkl(fn):
    """Read a FEKO instance from a pickle file"""
    fh=open(fn,'rb')
    feko=pkl.load(fh)
    fh.close()
    return feko

class FEKO():
    def __init__(self, fn, ftype='ascii'):
        if ftype.startswith('ascii'):
            fh=open(fn)
            d=fh.read()
            fh.close()
            fieldsStr=d.split('#Request')
            lines=fieldsStr[0].split('\n')
            #parse main header (fieldsStr[0])
            for l in lines:
                if l=='' or l.startswith('*'): continue
                elif l.startswith('#'):
                    #header information
                    hdr=l.split('#')[-1]
                    if hdr.lower().startswith('file type'): self.ftype=hdr.split(':')[-1].lstrip()
                    elif hdr.lower().startswith('file format'): self.fformat=hdr.split(':')[-1].lstrip()
                    elif hdr.lower().startswith('source'): self.source=hdr.split(':')[-1].lstrip()
                    elif hdr.lower().startswith('date'): self.date=hdr.split('e:')[-1].lstrip()
            self.fields=[]
            freqs=[]
            for field in fieldsStr[1:]:
                field=FEKOfield('#Request'+field)
                freqs.append(field.freq)
                self.fields.append(field)
            #sort field list by freq
            self.fields=[x for (y,x) in sorted(zip(freqs,self.fields))]
    def writeToFile(self, fn):
        """Write data to ASCII .ffe file"""
        fh=open(fn,'w')
        #main header
        ostr=''
        ostr+='##File Type: '+self.ftype+'\n'
        ostr+='##File Format: '+self.fformat+'\n'
        ostr+='##Source: '+self.source+'\n'
        ostr+='##Date: '+self.date+'\n'
        for f in self.fields:
            #write header and data
            ostr+='#Request Name: '+f.rname+'\n'
            ostr+='#Frequency: '+str(f.freq)+'\n'
            ostr+='#Coordinate System: '+f.coord+'\n'
            ostr+='No. of Theta Samples: '+str(f.stheta)+'\n'
            ostr+='No. of Phi Samples: '+str(f.sphi)+'\n'
            ostr+='#Result Type: '+f.rtype+'\n'
            ostr+='#No. of Header Lines: 1\n'
            ostr+='#                 \"Theta\"           \"Phi\"             \"Re(Etheta)\"      \"Im(Etheta)\"      \"Re(Ephi)\"        \"Im(Ephi)\"        \"Gain(Theta)\"     \"Gain(Phi)\"       \"Gain(Total)\"\n'
            for did in range(f.stheta*f.sphi):
                '                   %f   %f   %f   %f   %f   %f   %f   %f   %f\n'%(f.theta[did],f.phi[did],f.etheta[did].real,f.etheta[did].imag,f.ephi[did].real,f.ephi[did].imag,f.gtheta[did],f.gphi[did],f.gtotal[did])
        fh.close()

class FEKOfield():
    def __init__(self, d, ftype='ascii'):
        lines=d.split('\n')
        theta=[]
        phi=[]
        etheta=[]
        ephi=[]
        gtheta=[]
        gphi=[]
        gtotal=[]
        for l in lines:
            if l=='' or l.startswith('*'): continue
            elif l.startswith('#'):
                #header information
                hdr=l.split('#')[-1]
                if hdr.lower().startswith('request'): self.rname=hdr.split(':')[-1].lstrip()
                elif hdr.lower().startswith('freq'): self.freq=float(hdr.split(':')[-1].lstrip())
                elif hdr.lower().startswith('coord'): self.coord=hdr.split(':')[-1].lstrip()
                elif hdr.lower().startswith('no. of theta'): self.stheta=int(hdr.split(':')[-1].lstrip())
                elif hdr.lower().startswith('no. of phi'): self.sphi=int(hdr.split(':')[-1].lstrip())
                elif hdr.lower().startswith('result'): self.rtype=hdr.split(':')[-1].lstrip()
            else:
                #assume data format:
                #"Theta" "Phi" "Re(Etheta)" "Im(Etheta)" "Re(Ephi)" "Im(Ephi)" "Gain(Theta)" "Gain(Phi)" "Gain(Total)"
                cleanStr=" ".join(l.split())+" "
                dlist=map(float,cleanStr.split(' ')[:-1])
                theta.append(dlist[0])
                phi.append(dlist[1])
                etheta.append(dlist[2]+1j*dlist[3])
                ephi.append(dlist[4]+1j*dlist[5])
                gtheta.append(dlist[6])
                gphi.append(dlist[7])
                gtotal.append(dlist[8])
        self.theta=np.array(theta)
        self.phi=np.array(phi)
        self.etheta=np.array(etheta)
        self.ephi=np.array(ephi)
        self.gtheta=np.array(gtheta)
        self.gphi=np.array(gphi)
        self.gtotal=np.array(gtotal)

if __name__ == '__main__':
    print 'Running test cases...'

    print 'Reading FEKO file'
    fekoDataX=FEKO('/home/griffin/Downloads/PAPER_results/PAPER_FF_X.ffe')
    for f in fekoDataX.fields: print f.freq
    print 'Writing FEKO data to file'
    fekoDataX.writeToFile('test_feko.ffe')
    print 'Read back that text file to make sure it is correctly formatted'
    fekoDataXredux=FEKO('test_feko.ffe')
    print 'Writing FEKO instance to pickle file'
    writeFEKOPkl('test_feko.pkl',fekoDataX)
    print 'Reading FEKO instance from pickle file'
    fekoDataXreduxredux=readFEKOPkl('test_feko.pkl')

    print '...Made it through without errors'

