import numpy as np
def bhcoat(wavelength,radius_core,radius_mantle,ref_core,ref_mantle,ref_medium=1.0):
    """
    Python adaptation of the bhcoat code from the
    Bohren & Huffman 1983 light scattering text book.
    Calculates the extinction, absorption and refraction 
    efficiency factors (Qext, Qabs, Qref) for a given 
    coated dust grain.
    ---
    Input:
     wavelength = Wavelength that the calculation takes place in
     radius_core = radius of the dust grain core
     radius_mantle = radius of the icy mantle as measured from
                     the center of the dust grain
     ref_core = complex refractive index of the dust grain at
                the given wavelength
     ref_mantle = complex refractive index of the mantle at the
                  given wavelength
     ref_medium = realpart of the refractive index of the medium
                  at the given wavelength (assumed to be 1 for vacuum)
    Output:
     A python dictionary containing the keys 
     'Qsca' 'Qext' and 'Qback' which are the efficiency factors
     for scatterin, exctinction and backscattering.
     Not sure what Qback is.
    ---
    Conversion from f77 by Aleksi Suutarinen (aleksi.suutarinen@gmail.com)
    """
    eps = 1.0e-8 #Inner sphere convergence criterion
    #ii=[0.0,1.0] #Fortran imaginary number?
    #ii=np.complex(0,1)

    sizeparam_core=2*np.pi*radius_core*ref_medium/wavelength #x
    sizeparam_mantle=2*np.pi*radius_mantle*ref_medium/wavelength #y
    relref_core=ref_core/ref_medium #rfrel1
    relref_mantle=ref_mantle/ref_medium #rfrel2

    #Can't really give these descriptive names. They're 
    #just shorthand to be given to the functions
    m1x = relref_core*sizeparam_core #x1
    m2x = relref_mantle*sizeparam_core #x2
    m2y = relref_mantle*sizeparam_mantle #y2

    #Iteration stops after ystop terms
    nstop = np.ceil(sizeparam_mantle+4.*sizeparam_mantle**0.333 + 2.0)
    relref_mantlecore = relref_mantle/relref_core #refrel
    #Seriously, none of these really have any meaningful names from now on.
    #I'll just stop trying to give them any and stick to cleaning the code.
    D0_m1x = np.cos(m1x)/np.sin(m1x)
    D0_m2x = np.cos(m2x)/np.sin(m2x)
    D0_m2y = np.cos(m2y)/np.sin(m2y)
    psi0_y = np.cos(sizeparam_mantle)
    psi1_y = np.sin(sizeparam_mantle)
    chi0_y = -1.0*np.sin(sizeparam_mantle)
    chi1_y = np.cos(sizeparam_mantle)
    xi0_y = np.complex(psi0_y,-1.0*chi0_y)
    xi1_y = np.complex(psi1_y,-1.0*chi1_y)
    chi0_m2y = -1.0*np.sin(m2y)
    chi1_m2y = np.cos(m2y)
    chi0_m2x = -1.0*np.sin(m2x)
    chi1_m2x = np.cos(m2x)
    Qsca = 0.0
    Qext = 0.0
    Xback = np.complex(0.0,0.0)
    n = 1
    insphere_convergence = False

    for n in np.arange(1,nstop+1):
        rn = n
        psi_y = (2.0*rn-1.)*psi1_y/sizeparam_mantle - psi0_y
        chi_y = (2.0*rn-1.)*chi1_y/sizeparam_mantle - chi0_y
        xi_y = np.complex(psi_y,-1.0*chi_y)
        D1_m2y = 1.0/(rn/m2y-D0_m2y) - rn/m2y
        if not insphere_convergence:
            D1_m1x = 1.0/(rn/m1x-D0_m1x) - rn/m2x
            D1_m2x = 1.0/(rn/m2x-D0_m2x) - rn/m2x
            chi_m2x = (2.0*rn - 1.0)*chi1_m2x/m2x - chi0_m2x
            chi_m2y = (2.0*rn - 1.0)*chi1_m2y/m2y - chi0_m2y
            dchi_m2x = chi1_m2x - rn*chi_m2x/m2x #chipx2
            dchi_m2y = chi1_m2y - rn*chi_m2y/m2y #chipy2
            Ancap = relref_mantlecore*D1_m1x - D1_m2x
            Ancap = Ancap/(relref_mantlecore*D1_m1x*chi_m2x - dchi_m2x)
            Ancap = Ancap/(chi_m2x*D1_m2x - dchi_m2x)
            brack = Ancap*(chi_m2y*D1_m2y - dchi_m2y)
            Bncap = relref_mantlecore*D1_m2x - D1_m1x
            Bncap = Bncap/(relref_mantlecore*dchi_m2x - D1_m1x*chi_m2x)
            Bncap = Bncap/(chi_m2x*D1_m2x - dchi_m2x)
            crack = Bncap*(chi_m2y*D1_m2y - dchi_m2y)
            amess1 = brack*dchi_m2y
            amess2 = brack*chi_m2y
            amess3 = crack*dchi_m2y
            amess4 = crack*chi_m2y
            if np.abs(amess1) < eps*np.abs(D1_m2y) and \
               np.abs(amess2) < eps and \
               np.abs(amess3) < eps*np.abs(D1_m2y) and \
               np.abs(amess4) < eps:
                #Inner sphere convergence
                brack = np.complex(0.,0.)
                crack = np.complex(0.,0.)
                insphere_convergence = True
        Dnbar = D1_m2y - brack*dchi_m2y
        Dnbar = Dnbar/(1.0-brack*chi_m2y)
        Gnbar = D1_m2y - crack*dchi_m2y
        Gnbar = Gnbar/(1.0-crack*chi_m2y)
        an = (Dnbar/relref_mantle + rn/sizeparam_mantle)*psi_y - psi1_y
        an = an/((Dnbar/relref_mantle+rn/sizeparam_mantle)*xi_y-xi1_y)
        bn = (relref_mantle*Gnbar + rn/sizeparam_mantle)*psi_y - psi1_y
        bn = bn/((relref_mantle*Gnbar+rn/sizeparam_mantle)*xi_y-xi1_y)
        Qsca = Qsca + (2.0*rn+1.0)*(np.abs(an)*np.abs(an)+np.abs(bn)*np.abs(bn))
        Xback = Xback + (2.0*rn+1.0)*(-1.)**n*(an-bn)
        Qext = Qext + (2.0*rn + 1.0)*(np.real(an)+np.real(bn))
        psi0_y = psi1_y
        psi1_y = psi_y
        chi0_y = chi1_y
        chi1_y = chi_y
        xi1_y = np.complex(psi1_y,-1.0*chi1_y)
        chi0_m2x = chi1_m2x
        chi1_m2x = chi_m2x
        chi0_m2y = chi1_m2y
        chi1_m2y = chi_m2y
        D0_m1x = D1_m1x
        D0_m2x = D1_m2x
        D0_m2y = D1_m2y
    out_Qsca = (2.0/(sizeparam_mantle**2.0))*Qsca
    out_Qext = (2.0/(sizeparam_mantle**2.0))*Qext
    out_Qback = (np.abs(Xback))**2
    out_Qback = (1.0/(sizeparam_mantle**2.0))*out_Qback
    return {'Qsca':out_Qsca,'Qext':out_Qext,'Qback':out_Qback}
