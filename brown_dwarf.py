#from .tprofile import TemperatureProfile
#import numpy as np
#from taurex.data.fittable import fitparam
from taurex.util import movingaverage
from taurex.exceptions import InvalidModelException

from taurex.temperature import TemperatureProfile
from taurex.core import fitparam
import numpy as np



class BrownDwarf(TemperatureProfile):

    def __init__(self, T_top=200.0, P_surface=None,
                 P_top=None, deltaT_1=-50,deltaT_2=100 ,deltaT_3=100,deltaT_4=100,deltaT_5=100,
                 smoothing_window=10 ,limit_slope=9999999):
        
        super().__init__(self.__class__.__name__)
                
        #self._delta_t_points = delta_temperature_points
        self._deltaT_1 = deltaT_1
        self._deltaT_2 = deltaT_2
        self._deltaT_3 = deltaT_3
        self._deltaT_4 = deltaT_4
        self._deltaT_5 = deltaT_5
        self._p_points = [1e5, 1e4, 1e3, 1e1]
        self._T_top = T_top
        self._P_surface = P_surface
        self._P_top = P_top
        self._smooth_window = smoothing_window
        self._limit_slope = limit_slope
        
        #self.generate_pressure_fitting_params()
        #self.generate_temperature_fitting_params()
        

    @fitparam(param_name='T_top',
              param_latex='$T_\\mathrm{top}$',
              default_fit=False,
              default_bounds=[300, 2500])
    def temperatureTop(self):
        """Temperature at top of atmosphere in Kelvin"""
        return self._T_top

    @temperatureTop.setter
    def temperatureTop(self, value):
        self._T_top = value

    @fitparam(param_name='deltaT_1',
              param_latex='$\Delta_\\mathrm{T1}$',
              default_fit=False,
              default_bounds=[-500, 1e3])
    def deltaTemperature1(self):
        """ Difference between Temperature at top of atmosphere and temperature at 10Bar in Kelvin"""
        return self._deltaT_1

    @deltaTemperature1.setter
    def deltaTemperature1(self, value):
        self._deltaT_1 = value

    @fitparam(param_name='deltaT_2',
              param_latex='$\Delta_\\mathrm{T2}$',
              default_fit=False,
              default_bounds=[0, 1e3])
    def deltaTemperature2(self):
        """ Difference between Temperature at 10 bar and temperature at 1e3 bar in Kelvin"""
        return self._deltaT_2

    @deltaTemperature2.setter
    def deltaTemperature2(self, value):
        self._deltaT_2 = value
        
    @fitparam(param_name='deltaT_3',
              param_latex='$\Delta_\\mathrm{T3}$',
              default_fit=False,
              default_bounds=[0, 1e3])
    def deltaTemperature3(self):
        """ Difference between Temperature at 1e3 bar and temperature at 1e4 bar in Kelvin"""
        return self._deltaT_3

    @deltaTemperature3.setter
    def deltaTemperature3(self, value):
        self._deltaT_3= value
        
    @fitparam(param_name='deltaT_4',
              param_latex='$\Delta_\\mathrm{T4}$',
              default_fit=False,
              default_bounds=[0, 1e3])
    
    def deltaTemperature4(self):
        """ Difference between Temperature at 1e4 bar and temperature at 1e5 bar in Kelvin"""
        return self._deltaT_4

    @deltaTemperature4.setter
    def deltaTemperature4(self, value):
        self._deltaT_4 = value
        
    @fitparam(param_name='deltaT_5',
              param_latex='$\Delta_\\mathrm{T5}$',
              default_fit=False,
              default_bounds=[0, 1e3])
    def deltaTemperature5(self):
        """ Difference between Temperature at 1e5 bar and bottom of the atmosphere in Kelvin"""
        return self._deltaT_5

    @deltaTemperature5.setter
    def deltaTemperature5(self, value):
        self._deltaT_5 = value
        
    def check_profile(self, Ppt, Tpt):
        
        if(any(Ppt[i] <= Ppt[i + 1] for i in range(len(Ppt)-1))):
            self.warning('Temperature profile is not valid - a pressure point is inverted')
            raise InvalidTemperatureException

        if(any(abs((Tpt[i+1]-Tpt[i])/(np.log10(Ppt[i+1])-np.log10(Ppt[i]))) >= self._limit_slope for i in range(len(Ppt)-1))):
            self.warning('Temperature profile is not valid - profile slope too high')
            raise InvalidTemperatureException

    @property
    def profile(self):
        
        Tnodes = [self._T_top + self._deltaT_1 + self._deltaT_2 + self._deltaT_3 +self._deltaT_4 +self._deltaT_5, self._T_top + self._deltaT_1 + self._deltaT_2 + self._deltaT_3 +self._deltaT_4, self._T_top + self._deltaT_1 + self._deltaT_2 + self._deltaT_3, self._T_top + self._deltaT_1 + self._deltaT_2, self._T_top + self._deltaT_1, self._T_top]
    

        Psurface = self._P_surface
        if Psurface is None or Psurface < 0:
            Psurface = self.pressure_profile[0]

        Ptop = self._P_top
        if Ptop is None or Ptop < 0:
            Ptop = self.pressure_profile[-1]

        Pnodes = [Psurface, 1e5, 1e4, 1e3, 1e1, Ptop]

        self.check_profile(Pnodes, Tnodes)

        smooth_window = self._smooth_window

        if np.all(Tnodes == Tnodes[0]):
            return np.ones_like(self.pressure_profile)*Tnodes[0]

        TP = np.interp((np.log10(self.pressure_profile[::-1])),
                       np.log10(Pnodes[::-1]), Tnodes[::-1])
        
        # smoothing T-P profile
        wsize = int(self.nlayers*(smooth_window / 100.0))
        if (wsize % 2 == 0):
            wsize += 1
        TP_smooth = movingaverage(TP, wsize)
        border = np.int((len(TP) - len(TP_smooth))/2)

        foo = TP[::-1]
        if len(TP_smooth) == len(foo):
            foo = TP_smooth[::-1]
        else:
            foo[border:-border] = TP_smooth[::-1]

        return foo

    def write(self, output):
        temperature = super().write(output)

        #temperature.write_scalar('T_surface', self._T_surface)
        temperature.write_scalar('T_top', self._T_top)
        temperature.write_scalar('deltaT_1',self._deltaT_1)
        temperature.write_scalar('deltaT_2',self._deltaT_2)
        temperature.write_scalar('deltaT_3',self._deltaT_3)
        temperature.write_scalar('deltaT_4',self._deltaT_4)
        temperature.write_scalar('deltaT_5',self._deltaT_5)

        P_surface = self._P_surface
        P_top = self._P_top
        if not P_surface:
            P_surface = -1
        if not P_top:
            P_top = -1

        temperature.write_scalar('P_surface', P_surface)
        temperature.write_scalar('P_top', P_top)
        temperature.write_array('pressure_points', np.array(self._p_points))

        temperature.write_scalar('smoothing_window', self._smooth_window)

        return temperature

    @classmethod
    def input_keywords(cls):
        """
        Return all input keywords
        """
        return ['browndwarf', ]
