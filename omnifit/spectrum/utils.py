import numpy as np
import matplotlib.pyplot as plt

class baseliner:
  """
  baseliner of the objects
  """
  def __init__(self,ax,spec,power=1):
    self.ax = ax
    self.spec = spec
    self.x = spec.x
    self.y = spec.y
    self.power = power
    self.limlo=None
    self.limhi=None
    self.minx=np.min(self.x)
    self.maxx=np.max(self.x)
    self.miny=np.min(self.y)
    self.maxy=np.max(self.y)
    self.ax.set_xlim(self.minx,self.maxx)
    self.ax.set_ylim(self.miny,self.maxy)
    self.specplot,=self.ax.plot(self.x,self.y,'k-',drawstyle=spec.drawstyle)
    self.buttonListener = self.ax.figure.canvas.mpl_connect('button_press_event', self.mouse_press)
    self.keyListener = self.ax.figure.canvas.mpl_connect('key_press_event', self.key_press)
    self.windows=[]
  def key_press(self, event):
    if event.key=='q':
      self.skip()
    if event.key=='a' and self.limlo != None and self.limhi != None:
      self.addwindow(self.limlo,self.limhi)
      self.ax.plot([self.limlo,self.limlo],[self.miny,self.maxy],'g-')
      self.ax.plot([self.limhi,self.limhi],[self.miny,self.maxy],'g-')
      self.remlim()
      self.remlim()
      print 'Window added. Ready to receive another one.'
    else:
      return
  def mouse_press(self, event):
    if event.button==1:
      self.setlim(event.xdata)
    elif event.button==2:
      return
    elif event.button==3:
      self.remlim()
  def skip(self):
    plt.close()
  def setlim(self,i_x):
    if self.limlo==None:
      self.limlo=i_x
      self.limloplot,=self.ax.plot([i_x,i_x],[self.miny,self.maxy],'b-')
      self.ax.figure.canvas.draw()
    elif self.limhi==None:
      self.limhi=i_x
      self.limhiplot,=self.ax.plot([i_x,i_x],[self.miny,self.maxy],'b-')
      self.ax.figure.canvas.draw()
      print 'Ready for finalising. Press once more to do so, or press a to add another window.'
    else:
      self.finalise()
  def remlim(self):
    if self.limhi!=None:
      self.limhi=None
      self.limhiplot.set_ydata([self.miny,self.miny])
      self.ax.figure.canvas.draw()
    elif self.limlo!=None:
      self.limlo=None
      self.limloplot.set_ydata([self.miny,self.miny])
      self.ax.figure.canvas.draw()
    else:
      print 'No limits to cancel.'
  def addwindow(self,limlo,limhi):
    if limhi < limlo:
      limlo,limhi = limhi,limlo
    self.windows.append([limlo,limhi])
  def finalise(self):
    self.addwindow(self.limlo,self.limhi)
    self.ax.figure.canvas.mpl_disconnect(self.buttonListener)
    self.ax.figure.canvas.mpl_disconnect(self.keyListener)
    plt.close(self.ax.figure)

#--------------------------
#Misc. function definitions
#--------------------------
def wl2wn(iWavelength):
  """
  Convert wavelength [um] to wavenumber [cm^-1]
  """
  return 1.0e4/iWavelength

def wn2wl(iWavenumber):
  """
  Convert wavenumber [cm^-1] to wavelength [um]
  """
  return 1.0e4/iWavenumber

def cde_correct(wn,n,k):
  """
  CDE correction to  n,k data
  """
  wl=wn2wl(wn)
  m=np.vectorize(complex)(n,k)
  cabs_vol=np.empty(0)
  cabs=np.empty(0)
  cscat_vol=np.empty(0)
  for c_wn,c_wl,c_m in zip(wn,wl,m):
    m2=c_m**2.0
    im_part=((m2/(m2-1))*np.log(m2)).imag
    cabs_vol=np.hstack([cabs_vol,2.*(2.*np.pi/c_wl)*im_part])
    t_cabs=c_wn*(2.0*c_m.imag/(c_m.imag-1))*np.log10(c_m.imag)
    cabs=np.hstack([cabs,t_cabs])
    cscat_vol=np.hstack([cscat_vol,(c_wn**3.0/(6*np.pi))*t_cabs])
  ctot=cabs+cscat_vol
  return cabs,cabs_vol,cscat_vol,ctot
