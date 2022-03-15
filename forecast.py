#!/usr/bin/env python

import os
import sys
import argparse
import fitsio
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

def e(z,omegam) :
    return np.sqrt(omegam*(1+z)**3+(1-omegam))

def r(z,omegam) :
    assert(np.isscalar(z))
    H0=100 # km/s/Mpc
    cspeed=3e5 # km/s
    zz=np.linspace(0,z,1000)
    dz=zz[1:]-zz[:-1]
    zm=(zz[1:]+zz[:-1])/2.
    return np.sum(dz/e(zm,omegam))*cspeed/H0 # Mpc/h

def Plos_PD13(klos,zz,omegam=0.3):
    """A fit to the line-of-sight flux power spectrum, from
    Palanque-Delabrouille+(2013; https://arxiv.org/abs/1306.5896).
    This is Eq. (14) of that paper with the coefficients taken
    from the right-hand column of Table 3.
    Input:  klos line-of-sight wavevector in h/Mpc.
            zz   redshift.
    Output: Plos line-of-sight power spectrum in Mpc/h.
    """
    aH      = 100.0*e(zz,omegam)/(1+zz)
    k0,z0   = 0.009*aH,3.0
    AF      = 0.064
    nF,alphaF=-2.55,-0.10
    betaF,BF= -0.28,3.55
    krat,lnk= klos/k0,np.log(klos/k0)
    zrat    = (1+zz)/(1+z0)
    kPpi    = AF*(klos/k0)**(3+nF+alphaF*lnk+betaF*np.log(zrat))*zrat**BF
    if False: # Add Si contamination.
        a,deltav= 0.008/(1-FF),2271./aH
        kPpi   *= 1+a**2+2*a*np.cos(kfid*deltav)
    Plos    = kPpi*np.pi/klos
    return(Plos)

def Plya_DR12(z,k_camb,pk_camb,zref,k=0.15,mu=0.8,bias_lya=-0.122,beta_lya=1.66) :
    """Lya 3D power spectrum with RSD and damping
    Output: 3D power in (Mpc/h)^3
    """
    kd = 8.2 # h/Mpc
    Pmatter = np.interp(k,k_camb,pk_camb)
    Plya  = (bias_lya*(1+beta_lya*mu**2))**2 * Pmatter * np.exp(-(mu*k/kd)**2)
    return Plya * ((1+z)/(1+zref))**(2*(2.9-1)) # bias scaling as 2.9, growth rate= -1

lambda_lya = 1216.

def PNoise(snr,z,omegam=0.3) :
    """Noise power spectrum.
    Input:
      snr : signal to noise ratio, per sqrt(Angstrom)
      z   : redshift
      omegam : omegam
    Output: 1D noise power in (Mpc/h)
    """
    cspeed = 3e5
    H0 = 100.
    return 1./snr**2 * 1./lambda_lya * cspeed/(H0*e(z,omegam)) # Mpc/h


def main() :

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Approximate Fisher forecast for Lya auto-correlation")
    parser.add_argument("--snr-filename", type = str, default = "./lya-snr-fuji-sv3.fits", required = False,
                        help = 'SNR table filename, like lya-snr-fuji-sv3.fits, with columns Z,SNR (QSOTARGET)')
    parser.add_argument("--zbins", type = str, default = "2.12,2.28,2.43,2.59,2.75,2.91,3.07,3.23,3.39,3.55", required = False,
                        help = 'comma separated list of reshift bin centers, default="2.12,2.28,2.43,2.59,2.75,2.91,3.07,3.23,3.39,3.55"')
    parser.add_argument("--dndz", type = str, default = "96,81,65,52,40,30,22,16,11,7", required = False,
                        help = 'comma separated QSO densities /deg2/z for the redshift bin centers, default="96,81,65,52,40,30,22,16,11,7"')
    parser.add_argument("--pk-filename", type = str, default="./PlanckDR16.fits",required = False, help="Fiducial cosmology matter P(k) at zref")
    parser.add_argument("--with-non-qso-targets",action="store_true")
    parser.add_argument("-o","--outfile",type=str,default=None,help="output table with forecast results")
    parser.add_argument("--scale-dndz",type=float,default=1.,help="apply a scale factor to dndz")
    parser.add_argument("--rest-frame-min-wavelength",type=float,default=1040.,help="min wavelength of forest (rest-frame)")
    parser.add_argument("--rest-frame-max-wavelength",type=float,default=1200.,help="max wavelength of forest (rest-frame)")
    parser.add_argument("--observer-frame-min-wavelength",type=float,default=3600.,help="observer frame min wavelength")
    parser.add_argument("--bias-lya-at-zref",type=float,default=-0.122,help="lya bias at ZREF given in header of P(k) fits file")
    parser.add_argument("--beta-lya-at-zref",type=float,default=1.66,help="lya bias at ZREF given in header of P(k) fits file")

    args = parser.parse_args()

    # fiducial cosmo
    if not os.path.isfile(args.pk_filename) :
        print("Cannot open",args.pk_filename)
        print("Either set option --pk-filename or move to the directory where the default file is. Try also --help.")
        sys.exit(1)

    camb_table,camb_header=fitsio.read(args.pk_filename,header=True)
    print("P(k) header:")
    print(camb_header)
    omegam = camb_header["OM"]
    print("use omegam=",omegam)

    # read SNR
    if not os.path.isfile(args.snr_filename) :
        print("Cannot open",args.snr_filename)
        print("Either set option --snr-filename or move to the directory where the default file is. Try also --help.")
        sys.exit(1)


    tt=fitsio.read(args.snr_filename)

    # Get the QSO redshift, SNR and target flag.
    # We can apply any other cuts we want here (currently none).
    if 'QSOTARGET' in tt.dtype.names :
        if not args.with_non_qso_targets :
            print("Do not include non qso targets in SNR")
            tt=tt[tt['QSOTARGET']==1]
        else :
            print("Include non qso targets in SNR")
            pass # nothing to do

    zqso = tt['Z'].clip(1e-5,1e5)
    snr  = tt['SNR'].clip(1e-5,1e5)

    # the noise power as a function of S/N:
    PN = PNoise(snr,zqso,omegam=omegam)

    #  n(z) for Lya QSO
    zbins=np.array([float(z) for z in args.zbins.split(",")])
    dndz=np.array([float(dndz) for dndz in args.dndz.split(",")])
    if args.scale_dndz != 1. :
        print("Apply a scale factor to dndz=",args.scale_dndz)
        dndz *= args.scale_dndz

    print("zbins=",zbins)
    print("dndz=",dndz)

    dz=np.median(np.gradient(zbins))
    zedges=np.append(zbins-dz/2,zbins[-1]+dz/2)
    dz=(zedges[1:]-zedges[:-1])
    ntot = np.sum(dndz*dz)

    forecast_table=Table()
    forecast_table["z"]=zbins
    forecast_table["dndz_deg2z"]=dndz

    print("ntot= {:.2f}/deg2".format(ntot))

    # densities in redshift slice
    rr=np.array([r(z,omegam=omegam) for z in zbins])
    rr_edges=np.array([r(z,omegam=omegam) for z in zedges])
    drr=(rr_edges[1:]-rr_edges[:-1])
    dn = dndz*dz*(180/np.pi)**2/rr**2/drr # (Mpc/h)^-3

    if args.with_non_qso_targets and ( 'QSOTARGET' in tt.dtype.names ) :
        print("scale density to include non QSO targets")
        for i in range(zbins.size) :
            n1=float(np.sum((zqso>=zedges[i])&(zqso<zedges[i+1])))
            n2=float(np.sum((zqso>=zedges[i])&(zqso<zedges[i+1])&(tt["QSOTARGET"]==1)))
            scale=n1/n2
            print("at z={} scale factor applied to density to include non QSO targets = {}".format(zbins[i],scale))
            dn[i] *= scale

    forecast_table["density_qso_targets_hMpc3"]=dn

    # cosmological signal at fiducial k
    klos = 0.15 # A fiducial k-value for the QSO value, in h/Mpc.
    Plos = Plos_PD13(klos,zqso,omegam=omegam)

    # compute mean weight of quasar, nu, per redshift bin
    nu = np.zeros(zbins.size)
    for i in range(zbins.size) :
        selection=(zqso>=zedges[i])&(zqso<zedges[i+1])
        nu[i] = np.mean(Plos[selection]/(Plos[selection]+PN[selection]))

    forecast_table["nu_qso_targets"]=nu # no dim.

    # vol
    volume = rr**2*14000./(180/np.pi)**2 * drr # (Mpc/h)**3

    forecast_table["redshift_bin_depth_Mpch"]=drr
    forecast_table["volume_Mpch3"]=volume

    # forest length
    z1 = (1+zbins)*args.rest_frame_min_wavelength/lambda_lya-1.
    z2 = (1+zbins)*args.rest_frame_max_wavelength/lambda_lya-1.
    zmin=args.observer_frame_min_wavelength/lambda_lya-1
    z1[z1<zmin]=zmin
    z2[z2<zmin]=zmin
    r2=np.array([r(z,omegam=omegam) for z in z2])
    r1=np.array([r(z,omegam=omegam) for z in z1])
    length=r2-r1
    forecast_table["forest_length_Mpch"]=length # (Mpc/h)

    sigma_log_dh = np.zeros(zbins.size)
    sigma_log_da = np.zeros(zbins.size)
    corr_coef    = np.zeros(zbins.size)

    for iz in range(zbins.size) :
        z    = zbins[iz]
        neff = nu[iz]*dn[iz] # (Mpc/h)^{-3} , not quite the way to do it because nu contains Plos already ...
        lya_length = length[iz] # (Mpc/h)
        k   = np.linspace(0.01,1,1000) # h/Mpc
        dk  = (k[1]-k[0]) # h/Mpc
        dlk = dk/k
        lk  = np.log(k)

        fisher_matrix = np.zeros((2,2))
        mu_bin_edges  = np.linspace(0,1.,20)

        for mu_index in range(mu_bin_edges.size-1) :
            mu  = (mu_bin_edges[mu_index+1]+mu_bin_edges[mu_index])/2.
            dmu = mu_bin_edges[mu_index+1]-mu_bin_edges[mu_index]
            p3d = Plya_DR12(z=z,k_camb=camb_table["K"],pk_camb=camb_table["PK"],zref=camb_header["ZREF"],k=k,mu=mu,\
                            bias_lya=args.bias_lya_at_zref,
                            beta_lya=args.beta_lya_at_zref) # (Mpc/h)^{3}
            p1d = Plos_PD13(k*mu,z) # (Mpc/h)^{1}

            # total power
            p_tot = p3d + p1d/(neff*lya_length)

            # see Seo & Eisenstein, 2003 (Eq. 8), https://arxiv.org/pdf/astro-ph/0307460.pdf
            # compute number of independent modes
            number_of_modes = volume[iz] * k**2*dk*dmu / (2*np.pi**2) # = (4 pi k**2 dk dmu) / (2 pi)**3

            # variance of the measured power spectrum p3d
            var_p3d = 2 * p_tot**2 / number_of_modes  # (Mpc/h)^{6} , var(P) = 2*P**2/n

            # now do the wiggle fit along k
            # compute a smooth version of p3d
            # not sure how to do much better than a polynomial fit
            x=np.log(k)
            y=np.log(p3d)
            x -= np.mean(x)
            x /= (np.max(x)-np.min(x))
            w=np.ones(x.size)
            w[:3] *= 1.e8
            coef=np.polyfit(x,y,8,w=w)
            pol=np.poly1d(coef)
            smooth_p3d = np.exp(pol(x))
            wiggles      = p3d-smooth_p3d

            # add gaussian damping
            kp = mu*k
            kt = np.sqrt(1-mu**2)*k

            # Eisenstein, Seo, White, 2007, Eq. 12
            SigNLt = 3.26 # Mpc/h
            f      = 0.96 # logarithmic growth rate for omegam=0.3 z~2.3
            SigNLp = (1+f)*SigNLt # Mpc/h
            wiggles *= np.exp(-(SigNLp*kp)**2/2-(SigNLt*kt)**2/2)

            # derivative of model wrt to log(k)
            dwiggles     = np.zeros(k.size)
            dwiggles[1:] = wiggles[1:]-wiggles[:-1]
            dwigglesdlk  = dwiggles/dlk

            # k = sqrt( kp**2 + kt**2)
            #   = sqrt( ap**2*k**2*mu2 + at**2*k**2*(1-mu2))
            #   = k*sqrt( ap**2*mu2 + at**2*(1-mu2))
            # dk/dap         = mu2 * k
            # dlog(k)/dap    = mu2
            # dlog(k)/dat    = (1-mu2)
            # dmodel/dap     = dmodel/dlog(k)*dlog(k)/dap    = dmodeldlk * mu2
            # dmodel/dat     = dmodel/dlog(k)*dlog(k)/dat    = dmodeldlk * (1-mu2)
            h=[mu**2,(1-mu**2)]
            fisher_matrix += np.outer(h,h)*np.sum(dwigglesdlk**2/var_p3d)

        cov = np.linalg.inv(fisher_matrix)
        sigma_log_dh[iz] = np.sqrt(cov[0,0])
        sigma_log_da[iz] = np.sqrt(cov[1,1])
        corr_coef[iz]    = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])

        print(z,sigma_log_dh[iz],sigma_log_da[iz],corr_coef[iz])

    forecast_table["sigma_log_dh"]=sigma_log_dh
    forecast_table["sigma_log_da"]=sigma_log_da
    forecast_table["corr_coef"]=corr_coef

    plt.figure("sigma")
    for key in ["sigma_log_dh","sigma_log_da"] :
        plt.plot(zbins,forecast_table[key],label=key)
    plt.xlabel("Redshift")
    plt.ylabel("Uncertainty")
    plt.grid()


    sigma_log_dh_tot = 1./np.sqrt(np.sum(1/forecast_table["sigma_log_dh"]**2))
    sigma_log_da_tot = 1./np.sqrt(np.sum(1/forecast_table["sigma_log_da"]**2))
    print("sigma_log_dh_tot = ",sigma_log_dh_tot)
    print("sigma_log_da_tot = ",sigma_log_da_tot)

    if args.outfile is not None :
        forecast_table.write(args.outfile,overwrite=True)
        print("wrote",args.outfile)


    plt.show()

if __name__ == '__main__':
    main()
