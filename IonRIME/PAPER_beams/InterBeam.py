
import numpy as np
import ephem
import fmt
import scipy
from scipy import interpolate,spatial
import healpy as hp
import pylab as plt
import fitsUtil
import os,sys
import pyfits,pywcs

def genInterpBeam(freq,fits):
  beam_freqs = np.zeros((11),)
  beam_freqs[0] = 1e8
  for k in range(1,len(beam_freqs)):
    beam_freqs[k] = beam_freqs[k-1] + 1e7

  fekoX=fmt.FEKO('/home/nunhokee/diffuse_ems/beams/PAPER_FF_X.ffe')
  fekoY=fmt.FEKO('/home/nunhokee/diffuse_ems/beams/PAPER_FF_Y.ffe')   
  feko_xpol=fekoX.fields[0]
  feko_ypol=fekoY.fields[0]
  gxx = feko_xpol.etheta*np.conj(feko_xpol.etheta)+feko_xpol.ephi*np.conj(feko_xpol.ephi)
  beam = np.ndarray(shape=(gxx.shape[0],4,beam_freqs.shape[0]),dtype=complex)
   
  for j in range(0,len(beam_freqs)):
    feko_xpol = fekoX.fields[j]
    feko_ypol = fekoY.fields[j] 
    beam[:,0,j] = feko_xpol.etheta*np.conj(feko_xpol.etheta)+feko_xpol.ephi*np.conj(feko_xpol.ephi)
    beam[:,0,j] = beam[:,0,j] / np.max(beam[:,0,j])						
    beam[:,1,j] = feko_xpol.etheta*np.conj(feko_ypol.etheta)+feko_xpol.ephi*np.conj(feko_ypol.ephi)
    beam[:,2,j] = feko_ypol.etheta*np.conj(feko_xpol.etheta)+feko_ypol.ephi*np.conj(feko_xpol.ephi)
    beam[:,3,j] = feko_ypol.etheta*np.conj(feko_ypol.etheta)+feko_ypol.ephi*np.conj(feko_ypol.ephi)
    beam[:,3,j] = beam[:,3,j] / np.max(beam[:,3,j])	    
  
  
  im,hdr,axisInfo=fitsUtil.readFITS(fits,hdr=True,axis=True) #load template FITS
  wcs=hdr['wcs']
  axisDict={}
  for aid,aname in enumerate(axisInfo):
    if aname.startswith('RA'): axisDict['RA']=aid
    if aname.startswith('DEC'): axisDict['DEC']=aid
    if aname.startswith('FREQ'): axisDict['FREQ']=aid
    if aname.startswith('STOKES'): axisDict['STOKES']=aid

  #(l,m) grid for spatial interpolation
  phi=fekoX.fields[0].phi*np.pi/180.
  theta=fekoX.fields[0].theta*np.pi/180.
  #drop beam points near zenith as this can create artefacts
  deltaTheta=(np.pi/2.)/100. #delta of theta away from pi/2 to ignore
  thetaIdx=np.argwhere(theta<(np.pi/2.-deltaTheta))
  
  theta0=theta.flatten()
  phi0=phi.flatten()
  #Generate a WCS structure with the same resolution as the template FITS file, but centered at the North Pole
  wcs = pywcs.WCS(naxis=2)
  wcs.wcs.crval = [0.,90.]
  wcs.wcs.crpix = [hdr['raPix'],hdr['decPix']]
  wcs.wcs.cdelt = [hdr['dra'],hdr['ddec']]
  wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
  #compute the sky coordinates for each pixel
  mms=np.repeat(np.arange(im.shape[axisDict['RA']]),im.shape[axisDict['DEC']]).reshape((im.shape[axisDict['RA']],im.shape[axisDict['DEC']]))
  lls=mms.T
  imPhi,imTheta=wcs.wcs_pix2sky(np.array(mms.flatten(), np.float_),np.array(lls.flatten(), np.float_),1)  #compute phi(RA)/theta(DEC)
  imPhi=imPhi.reshape(im.shape[axisDict['RA']],im.shape[axisDict['DEC']])*np.pi/180.                                 
  imTheta-=90.
  imTheta*=-1.
  imTheta=imTheta.reshape(im.shape[axisDict['RA']],im.shape[axisDict['DEC']])*np.pi/180.                             
  InterpBeamIm=np.zeros((len(polID),im.shape[axisDict['RA']],im.shape[axisDict['DEC']]),dtype=complex) 
  freqInterpBeams={}
  print "Frequency Interpolation"
  InterBeamF = np.zeros(shape=(gxx.shape[0],4),dtype=complex)
  df = np.abs(beam_freqs - freq)
  ind = np.argsort(df)
  for pid,plabel in enumerate(polID):
    if freq==beam_freqs[ind[0]]:
       InterBeamF[:,pid] = beam[:,pid,ind[0]]    
    else:
       InterBeamF[:,pid] = (beam[:,pid,ind[0]]* df[ind[0]]**(-2.) + beam[:,pid,ind[1]]* df[ind[1]]**(-2.))/(df[ind[0]]**(-2.)+df[ind[1]]**(-2.))
 
    InterBeamF0=InterBeamF[:,pid].flatten() 
    freqInterpBeams[plabel]=InterBeamF0
   
  print "Spatial Interpolation" 
  for pid,plabel in enumerate(polID):
    InterpBeamIm0 = interpolate.griddata(np.column_stack((phi0, theta0)), freqInterpBeams[plabel],np.column_stack((imPhi.flatten(),imTheta.flatten())),method='linear') 
    InterpBeamIm[pid]=InterpBeamIm0.reshape(imPhi.shape)
      
  return InterpBeamIm


if __name__ == '__main__':
    from optparse import OptionParser
    o = OptionParser()
    o.set_usage('%prog [options] STOKESIQUV COMBINED IMAGE')
    o.add_option('-o','--ofn',dest='ofn',default=None,
        help='Output filename of numpy file, default: beam_[FREQ]_[RES].npy')
    o.add_option('-o','--ofits',dest='ofits',default=None,
        help='Output filename of fits file, default: beam_[FREQ]_[RES].fits')
    opts, args = o.parse_args(sys.argv[1:])
   
    
    fn=args[0]
    hdu = pyfits.open(fn)
    hdr = hdu[0].header
    naxis = hdr['NAXIS1']
    freq = hdr['CRVAL3']
    res = hdr['CDELT1']
    data = hdu[0].data
    data = data[:,0,:,:]
    stokes=.5*np.matrix([[1.,1.,0.,0.],[0.,0.,1.,1.j],[0.,0.,1.,-1.j],[1.,-1.,0.,0.]])
    
    jones = np.zeros((2,2,naxis,naxis),dtype=complex)
    outjones = np.zeros((4,4,naxis,naxis),dtype=complex)
    outjones_inv = np.zeros((4,4,naxis,naxis),dtype=complex)
    sinverse = np.zeros((4,4,naxis,naxis),dtype=complex)
    sjones = np.zeros((4,4,naxis,naxis),dtype=complex)
    corr_data = np.zeros((4,naxis,naxis),dtype=complex)
    beam=genInterpBeam(freq,fn)  
    jones[0,0,:,:]=beam[0,:,:] 
    jones[0,1,:,:]=beam[1,:,:] 
    jones[1,0,:,:]=beam[2,:,:] 
    jones[1,1,:,:]=beam[3,:,:] 
    for i in range(naxis):
      for j in range(naxis):
         outjones[:,:,i,j] = np.kron(jones[:,:,i,j],jones[:,:,i,j].conj().T)
         outjones_inv[:,:,i,j] = np.linalg.inv(outjones[:,:,i,j])        
         sinverse[:,:,i,j] = np.dot(np.array(np.linalg.inv(stokes)),outjones_inv[:,:,i,j]) 
         sjones[:,:,i,j] = np.dot(sinverse[:,:,i,j],stokes)
         corr_data[:,i,j] = np.dot(sjones[:,:,i,j],data[:,i,j])
    
    path,srcFn=os.path.split(os.path.realpath(fn))
    if opts.ofn is None:
        dstFn='beam_f%.2f_r%i%s%s.npy'%(freq*10e-7,int(np.abs(res*3600.)))
    else: dstFn=opts.ofn
    print 'Creating numpy file:',os.path.join(path,dstFn)
    np.save(os.path.join(path,dstFn),jones=jones,outjones=outjones,outjones_inv=outjones_inv,sjones=sjones,corr_data=corr_data)
    
    if opts.ofits is None:
        dstFits='beam_f%.2f_r%i%s%s.fits'%(freq*10e-7,int(np.abs(res*3600.)))
    else: dstFits=opts.ofits
    print "Creating beam corrected fits",os.path.join(path,dstFits)
    pyfits.writeto(filename,corr_data.real,hdr,clobber=True) 
